from anomalydetector.multidata_handler import MultiDataHandler
from anomalydetector.utils import (
                    full_signal,
                    reconstruct_signal,
                    form_xy,
                    reshape_for_cheb,
                    make_cheb_basis,
                    optimal_weighted_regression,
                    )
from spcqe.quantiles import SmoothPeriodicQuantiles

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import warnings
import joblib
import os
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


# =================================================================================================
# ===============================   OUTAGEPIPELINE CLASS   ========================================
# =================================================================================================


class OutagePipeline:
    def __init__(self,
                 sites,
                 ndil,
                 target = None,
                 weight_quantiles = 5,
                 quantiles = None,
                 solver_quantiles = 'clarabel',
                 num_harmonics = [30,3],
                 nlag = 3,
                 num_basis = 8,
                 weight_linear = None,
                 lambda_range = np.logspace(0, 3, num=10),
                 num_split = 5,
                 model_residuals = 'ensemble',
                 train_size = 0.8
                 ):
        """
        This class contains the object QLinear which is the superposition of a quantile estimation, linear model and 
        binary classification. The object contains the 3 models and the intermediate results for the train set and test set.
        :param sites: the list of the name of the sites
        :type sites: str list

        :param ndil: the number of steps by days, if the dilution of the MultidataHandler hasn't been done the fit/predict 
        start by diluating the data.
        :type ndil: int, optional

        :param target: the name of the target, if the failure scenario generation hasn't been made the fit function start by 
        generating a failure scenarion with default parameters.
        :type target: str, optional
        """
        self.sites = sites
        self.ndil = ndil
        if target is None :
            self.target = self.sites[0]
        else :
            self.target = target
        self.weight_quantiles = weight_quantiles
        if quantiles is None :
            self.quantiles = np.array([0.02, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.98])
        else :
            self.quantiles = quantiles
        self.solver_quantiles = solver_quantiles
        self.num_harmonics = num_harmonics
        if weight_linear is None :
            n_features = (len(self.sites)-1) * (2*nlag+1) + 1
            _default_weight = np.tile(np.arange(1, num_basis+1)**2,n_features)
            self.weight_linear = _default_weight
        else :
            self.weight_linear = weight_linear
        self.lambda_range = lambda_range
        self.num_split = num_split
        

        if isinstance(model_residuals,str) :
            opt_n_default = 57
            if model_residuals == "SVM":
                from sklearn.svm import SVC
                svm_model = SVC(kernel='rbf')
                self.residual_model = svm_model
            if model_residuals == 'ensemble':
                try : 
                    from sklearn.svm import SVC
                    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.ensemble import VotingClassifier
                    from xgboost import XGBClassifier
                    from sklearn.decomposition import PCA
                    from sklearn.pipeline import Pipeline
                except ModuleNotFoundError:
                    warnings.warn("xgboost is not installed. Some features may be unavailable.")
                    self.residual_model = None
                else :
                    xgboost_ensemble = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
                    svm_ensemble = SVC(probability=True)  
                    pca_qda = Pipeline([
                    ('pca', PCA(n_components=opt_n_default)), 
                    ('qda', QuadraticDiscriminantAnalysis())
                    ])
                    pca_logreg = Pipeline([
                    ('pca', PCA(n_components=opt_n_default)),
                    ('logreg', LogisticRegression())
                    ])
                    ensemble_model = VotingClassifier(
                        estimators=[
                            ('xgb', xgboost_ensemble),
                            ('svm', svm_ensemble),
                            ('PCAlogreg', pca_logreg),
                            ('PCA_QDA', pca_qda)
                        ],
                        voting='soft'
                    )
                    self.residual_model = ensemble_model
            if model_residuals == 'logisticRegression':
                from sklearn.linear_model import LogisticRegression
                logreg_model = LogisticRegression(max_iter=1000)
                self.residual_model = logreg_model
            if model_residuals == 'XGBoost' :
                try :
                    from xgboost import XGBClassifier
                except ModuleNotFoundError:
                    warnings.warn("xgboost is not installed. Some features may be unavailable.")
                    self.residual_model = None
                else :
                    xgboost_model = XGBClassifier(
                                    objective="binary:logistic",
                                    eval_metric="logloss",
                                    max_depth=10,
                                    learning_rate=0.05,      
                                    n_estimators=100,       
                                    early_stopping_rounds=10,
                                    use_label_encoder=False  
                                )
                    self.residual_model = xgboost_model
            if model_residuals == 'QDA' :
                from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
                qda_model = QuadraticDiscriminantAnalysis()
                self.residual_model = qda_model
            if model_residuals == 'LDA' :
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                lda_model = LinearDiscriminantAnalysis()
                self.residual_model = lda_model
            if model_residuals == 'PCA+LDA' :
                from sklearn.decomposition import PCA
                from sklearn.pipeline import Pipeline
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                pca_lda = Pipeline([
                    ('pca', PCA(n_components=opt_n_default)),
                    ('lda', LinearDiscriminantAnalysis())
                    ])
                self.residual_model = pca_lda
            if model_residuals == 'PCA+QDA' :
                from sklearn.decomposition import PCA
                from sklearn.pipeline import Pipeline
                from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
                pca_lda = Pipeline([
                    ('pca', PCA(n_components=opt_n_default)),
                    ('qda', QuadraticDiscriminantAnalysis())
                    ])
                self.residual_model = pca_qda
            if model_residuals == 'PCA+logisticRegression' :
                from sklearn.decomposition import PCA
                from sklearn.pipeline import Pipeline
                from sklearn.linear_model import LogisticRegression
                pca_lda = Pipeline([
                    ('pca', PCA(n_components=opt_n_default)),
                    ('qda', LogisticRegression())
                    ])
                self.residual_model = pca_logreg
            else :
                self.residual_model = None
        else :
            self.residual_model = model_residuals
        self.train_size = train_size
        self.nlag = nlag
        self.num_basis = num_basis
        self._initialize_attributes()

    def _initialize_attributes(self):
        self.spqs = None
        self.start_train = None
        self.quantile_train = None
        self.quantile_test = None
        self.quantile_failure = None
        self.quantile_failure_test = None
        self.residuals_train = None
        self.residuals_test = None
        self.residuals_failure = None
        self.residuals_failure_test = None
        self.linear_coeff = None
        self.preprocess = None

    def fit_quantiles(self,
                      multidata,
                      weight_quantiles = None,
                      quantiles = None,
                      solver_quantiles = None,
                      num_harmonics = None,
                      ):
        """
        This function fits the first part of the model and compute the intermediate results for the training set
        :param multidata: This object contains all information about the sites, if the data haven't been diluated yet 
        and self.ndil is given the function start by diluating the data. 
        :type multidata: MultiDataHandler
        :param param: Dictionnary of every possible parameters for the learning step: weight_quantiles, quantiles, 
        solver_quantiles, num_harmonics,client.
        :type param: dict
        """

        if multidata.dil_mat is not None and self.ndil is not None and multidata.ndil() != self.ndil:
            raise ValueError("ndil between the argument and the model is different")
        elif multidata.dil_mat is None and self.ndil is not None :
            multidata.dilate(ndil = self.ndil)
        elif multidata.dil_mat is not None and self.ndil is None :
            self.ndil = multidata.ndil()
        elif multidata.dil_mat is None and self.ndil is None :
            raise ValueError('No ndil given')
        
        if multidata.target is not None and self.target is not None and multidata.target != self.target:
            raise ValueError("target between the argument and the model is different")
        elif multidata.target is None and self.target is not None :
            multidata.generate_failure(self.target)
            warnings.warn(
                "No failure scenario detected — calling multidata.generate_failure automatically with default parameters.",
                category=UserWarning
            )
        elif multidata.target is not None and self.target is None :
            self.target = multidata.target
        elif multidata.target is None and self.target is None :
            raise ValueError('No target given')
        
        self.start_train = multidata.common_days[0]
        self.weight_quantiles = weight_quantiles if weight_quantiles is not None else self.weight_quantiles
        self.quantiles = quantiles if quantiles is not None else self.quantiles
        self.solver_quantiles = solver_quantiles if solver_quantiles is not None else self.solver_quantiles 
        self.num_harmonics = num_harmonics if num_harmonics is not None else self.num_harmonics 

        dict_quantile = {}
        for site in self.sites :
            dict_quantile[site] = SmoothPeriodicQuantiles(
            num_harmonics=self.num_harmonics,
            periods=[self.ndil, 365.24225*self.ndil],
            standing_wave=[True, False],
            trend=False,
            quantiles=self.quantiles,
            weight=self.weight_quantiles,
            problem='sequential',
            solver=self.solver_quantiles,
            extrapolate='solar'
            )
        self.spqs = dict_quantile

        self.quantile_train = {}
        for key in multidata.dil_mat :
            full_array,_ = full_signal(multidata.common_days,self.start_train,multidata.dil_mat[key],self.ndil)
            self.spqs[key].fit(full_array)
            self.quantile_train[key] = reconstruct_signal(self.spqs[key].transform(full_array),multidata.common_days,self.ndil)
        full_failure,_ = full_signal(multidata.common_days,self.start_train,multidata.failure_mat,self.ndil)
        self.quantile_failure = reconstruct_signal(self.spqs[self.target].transform(full_failure),multidata.common_days,self.ndil)

    def fit_linear(self,
                   nlag = None,
                   num_basis = None,
                   weight_linear = None,
                   lambda_range = None,
                   num_split = None
                   ):
        """
        This function fit the linear regression part. It creates some lagged features and learn a linear regression where the coefficients
        between each step in a day is driven by a chebyvev basis.
        :param param: Dictionnary of every possible parameters for the learning step: nlag, num_basis, weight_linear
        lambda_range, num_split.
        :type param: dict
        """
        X = np.array([self.quantile_train[key] for key in self.quantile_train if key != self.target])
        y = self.quantile_train[self.target]
        y_failure = self.quantile_failure
        #Numerical stability
        X = np.clip(X,-4,4)
        y = np.clip(y,-4,4)
        y_failure = np.clip(y_failure,-4,4)
        X[np.isnan(X)] = 0
        y[np.isnan(y)] = 0
        y_failure[np.isnan(y_failure)] = 0

        self.nlag = nlag if nlag is not None else self.nlag
        self.num_basis = num_basis if num_basis is not None else self.num_basis
        self.weight_linear = weight_linear if weight_linear is not None else self.weight_linear
        self.lambda_range = lambda_range if lambda_range is not None else self.lambda_range 
        self.num_split = num_split if num_split is not None else self.num_split

        X,y,y_failure = form_xy(X,y,y_failure,self.nlag)
        X_flat,y_flat = reshape_for_cheb(X,y,self.num_basis)
        _,y_failure_flat = reshape_for_cheb(X,y_failure,self.num_basis)
        self.lambda_optimal,self.linear_coeff = optimal_weighted_regression(X_flat,
                                                                            y_flat,
                                                                            self.weight_linear,
                                                                            self.lambda_range,
                                                                            n_splits=self.num_split)
        self.residuals_train = (y_flat-X_flat@self.linear_coeff).reshape(y.shape)
        self.residuals_failure = (y_failure_flat-X_flat@self.linear_coeff).reshape(y_failure.shape)

    def fit_residuals(self,
                      residual_model = None,
                      train_size = None
                      ):
        """
        Finaly this function takes the residuals computed in the prevous step and classify the failure and no-failure case.
        :param param: Dictionnary of every possible parameters for the learning step: model_residuals, train_size.
        :type param: dict
        """
        self.residual_model = residual_model if residual_model is not None else self.residual_model 
        self.train_size = train_size if train_size is not None else self.train_size 
        X = np.concat([self.residuals_train,self.residuals_failure])
        y = np.concat([[0]*self.residuals_train.shape[0],[1]*self.residuals_failure.shape[0]])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1-self.train_size, shuffle=True,random_state=0)

        self.preprocess = StandardScaler()
        self.preprocess.fit(X_train)
        X_train = self.preprocess.transform(X_train)
        X_val = self.preprocess.transform(X_val)

        self.residual_model.fit(X_train,y_train)

        return {"accuracy_train" : accuracy_score(y_train,self.residual_model.predict(X_train)),
                "f1_score_train" : f1_score(y_train,self.residual_model.predict(X_train)),
                "accuracy_val" : accuracy_score(y_val,self.residual_model.predict(X_val)),
                "f1_score_val" : f1_score(y_val,self.residual_model.predict(X_val))
                }

    def fit(self,
            mutlidata,
            weight_quantiles = None,
            quantiles = None,
            solver_quantiles = None,
            num_harmonics = None,
            client = None,
            nlag = None,
            num_basis = None,
            weight_linear = None,
            lambda_range = None,
            num_split = None,
            residual_model = None,
            train_size = None
            ):
        self.fit_quantiles(mutlidata,
                           weight_quantiles=weight_quantiles,
                           quantiles=quantiles,
                           solver_quantiles=solver_quantiles,
                           num_harmonics=num_harmonics,
                           client=client)
        self.fit_linear(nlag = nlag,
                        num_basis=num_basis,
                        weight_linear=weight_linear,
                        lambda_range=lambda_range,
                        num_split=num_split)
        return self.fit_residuals(residual_model=residual_model,
                                  train_size=train_size)
    
    def predict_quantiles(self,multidata):

        """
        This function computes the transform data and load the intermediate result for the test set.
        :param multidata: This object contains all information about the sites, if the data haven't been diluated yet 
        and self.ndil is given the function start by diluating the data. 
        :type multidata: MultiDataHandler
        :param param: Dictionnary of every possible parameters for the learning step: weight_quantiles, quantiles, 
        solver_quantiles, num_harmonics.
        :type param: dict
        """

        if multidata.dil_mat is not None and self.ndil is not None and multidata.ndil() != self.ndil:
            raise ValueError("ndil between the argument and the model is different")
        elif multidata.dil_mat is None and self.ndil is not None :
            multidata.dilate(ndil = self.ndil)
        elif multidata.dil_mat is not None and self.ndil is None :
            self.ndil = multidata.ndil()
        elif multidata.dil_mat is None and self.ndil is None :
            raise ValueError('No ndil given')
        
        if multidata.target is not None and self.target is not None and multidata.target != self.target:
            raise ValueError("target between the argument and the model is different")
        elif multidata.target is None and self.target is None :
            raise ValueError('No target given')
        
        for key in multidata.dil_mat :
            full_array,start_index = full_signal(multidata.common_days,self.start_train,multidata.dil_mat[key],self.ndil)
            time_index = np.arange(start_index, len(full_array) + start_index)
            self.quantile_test[key] = reconstruct_signal(self.spqs[key].transform(full_array,y = time_index),multidata.common_days,self.ndil)

    def predict_linear(self):
        """
        This function compute the linear regression part. It creates the lagged features and compute the residuals for the test set.
        :param param: Dictionnary of every possible parameters for the learning step: nlag, num_basis, weight_linear
        lambda_range, num_split.
        :type param: dict
        """
        X = np.array([self.quantile_test[key] for key in self.quantile_train if key != self.target])
        y = self.quantile_test[self.target]
        #Numerical stability
        X = np.clip(X,-4,4)
        y = np.clip(y,-4,4)
        X[np.isnan(X)] = 0
        y[np.isnan(y)] = 0

        X,y,_ = form_xy(X,y,y,self.nlag)
        X_flat,y_flat = reshape_for_cheb(X,y,self.num_basis)
        self.residuals_test = (y_flat-X_flat@self.linear_coeff).reshape(y.shape)

    def predict_residuals(self):
        """
        Finaly this function takes the residuals computed in the prevous step and predict the different case.
        :param param: Dictionnary of every possible parameters for the learning step: model_residuals, train_size.
        :type param: dict
        """
        X = np.concat([self.residuals_train,self.residuals_failure])
        X = self.preprocess.transform(X)
        y = self.residual_model.predict(X)
        return y

    def predict(self,multidata):
        self.predict_quantiles(multidata)
        self.predict_linear()
        return self.predict_residuals()

    def test_quantiles(self,multidata):
        """
        This function compute the first part of the model and compute the intermediate results for the test set and failure scenario
        :param multidata: This object contains all information about the sites, if the data haven't been diluated yet 
        and self.ndil is given the function start by diluating the data. 
        :type multidata: MultiDataHandler
        """

        if multidata.dil_mat is not None and self.ndil is not None and multidata.ndil() != self.ndil:
            raise ValueError("ndil between the argument and the model is different")
        elif multidata.dil_mat is None and self.ndil is not None :
            multidata.dilate(ndil = self.ndil)
        elif multidata.dil_mat is not None and self.ndil is None :
            self.ndil = multidata.ndil()
        elif multidata.dil_mat is None and self.ndil is None :
            raise ValueError('No ndil given')
        
        if multidata.target is not None and self.target is not None and multidata.target != self.target:
            raise ValueError("target between the argument and the model is different")
        elif multidata.target is None and self.target is not None :
            multidata.generate_failure(self.target)
            warnings.warn(
                "No failure scenario detected — calling multidata.generate_failure automatically with default parameters.",
                category=UserWarning
            )
        elif multidata.target is not None and self.target is None :
            self.target = multidata.target
        elif multidata.target is None and self.target is None :
            raise ValueError('No target given')
        
        self.quantile_test = {}
        for key in multidata.dil_mat :
            full_array,start_index = full_signal(multidata.common_days,self.start_train,multidata.dil_mat[key],self.ndil)
            time_index = np.arange(start_index, len(full_array) + start_index)
            self.quantile_test[key] = reconstruct_signal(self.spqs[key].transform(full_array,y = time_index),multidata.common_days,self.ndil)
        full_failure,start_index = full_signal(multidata.common_days,self.start_train,multidata.failure_mat,self.ndil)
        time_index = np.arange(start_index, len(full_array) + start_index)
        self.quantile_failure_test = reconstruct_signal(self.spqs[self.target].transform(full_failure,y = time_index),multidata.common_days,self.ndil)

    def test_linear(self):
        """
        This function computes the linear regression part. 
        It creates the lagged features and compute the residuals for the failure/nofailure scenrario.
        """
        X = np.array([self.quantile_test[key] for key in self.quantile_test if key != self.target])
        y = self.quantile_test[self.target]
        y_failure = self.quantile_failure_test
        #Numerical stability
        X = np.clip(X,-4,4)
        y = np.clip(y,-4,4)
        y_failure = np.clip(y_failure,-4,4)
        X[np.isnan(X)] = 0
        y[np.isnan(y)] = 0
        y_failure[np.isnan(y_failure)] = 0

        X,y,y_failure = form_xy(X,y,y_failure,self.nlag)
        X_flat,y_flat = reshape_for_cheb(X,y,self.num_basis)
        _,y_failure_flat = reshape_for_cheb(X,y_failure,self.num_basis)
        self.residuals_test = (y_flat-X_flat@self.linear_coeff).reshape(y.shape)
        self.residuals_failure_test = (y_failure_flat-X_flat@self.linear_coeff).reshape(y_failure.shape)

    def test_resiudals(self):
        """
        Finaly this function takes the residuals computed in the prevous step and classify the failure and no-failure case.
        """
        X = np.concat([self.residuals_test,self.residuals_failure_test])
        y = np.concat([[0]*self.residuals_test.shape[0],[1]*self.residuals_failure_test.shape[0]])

        X = self.preprocess.transform(X)

        return {"accuracy_test" : accuracy_score(y,self.residual_model.predict(X)),
                "f1_score_test" : f1_score(y,self.residual_model.predict(X))}

    def test(self,multidata):
        self.test_quantiles(multidata)
        self.test_linear()
        return self.test_resiudals()

    def display(self,idx, multidata = None):
        if multidata is None :
            _, ax = plt.subplots(figsize=(10, 4))

            residual_nofail = self.residuals_train[idx,:]
            residual_fail = self.residuals_failure[idx,:]
            x = np.arange(residual_fail.shape[0])
            
            ax.plot(x, residual_fail, label="Residuals with outage", color="blue")
            ax.plot(x, residual_nofail, label="Residuals no outage",
                    color="blue", linestyle="dotted", linewidth=1)
            ax.set_ylabel("Power")
            lines1, labels1 = ax.get_legend_handles_labels()
            ax.legend(lines1, labels1, loc="upper right")
            ax.grid(True)
            plt.tight_layout()
            plt.show()
        
        else :
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})

            residual_nofail = self.residuals_train[idx, :]
            residual_fail = self.residuals_failure[idx, :]
            x = np.arange(self.nlag, self.ndil - self.nlag-1)

            ax2.plot(x, residual_fail, label="Residuals with outage", color="blue")
            ax2.plot(x, residual_nofail, label="Residuals no outage", color="blue", linestyle="dotted", linewidth=1)

            ax2.set_ylabel("Power")
            ax2.legend(loc="upper right")

            multidata.display(idx,site = self.target,ax = ax1)
            ax2.grid(True)
            plt.tight_layout()
            plt.show()

    def coef_linear(self):
        nval = self.residuals_train.shape[1]
        c = make_cheb_basis(np.arange(nval), self.num_basis, (0, nval))
        theta_reshaped = self.linear_coeff.reshape(self.linear_coeff.shape[0]//self.num_basis, self.num_basis)
        coeffs = np.dot(c, theta_reshaped.T)
        return coeffs

    def display_quantiles(self, site=None, num_day=7, idx=0, print_train = True,figure = 'all'):
        if site is None:
            site = self.target
        if figure == 'all':
            fig = plt.figure(figsize=(18, 10))
            gs  = GridSpec(2, 2, height_ratios=[1, 0.8], figure=fig)

            # axe du haut sur toute la largeur
            ax_quant = fig.add_subplot(gs[0, :])
            # deux axes en bas
            ax_qq    = fig.add_subplot(gs[1, 0])
            ax_pp    = fig.add_subplot(gs[1, 1])

            # =======================
            #  Quantile functions
            # =======================
            cmap1 = plt.get_cmap('coolwarm')
            fq = self.spqs[site].fit_quantiles[idx * self.ndil:(idx + num_day) * self.ndil, :]
            colors1 = cmap1(np.linspace(0, 1, fq.shape[1]))

            for j in range(fq.shape[1]):
                ax_quant.plot(fq[:, j], color=colors1[j], alpha=0.5, label=self.quantiles[j])

            xticks_pos = np.arange(0, fq.shape[0], self.ndil)
            start_date = pd.to_datetime(self.start_train) + pd.to_timedelta(idx, unit="D")
            xticks_labels = [
                (start_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(num_day)
            ]
            ax_quant.set_xticks(xticks_pos)
            ax_quant.set_xticklabels(xticks_labels, rotation=45)
            ax_quant.set_title("Quantile functions")
            ax_quant.legend()

            # =======================
            #  QQ plot
            # =======================
            raw = self.quantile_train[site] if print_train else self.quantile_test[site]
            data = raw.T.flatten()

            osm, _ = stats.probplot(data, dist="norm")
            idxs = np.arange(len(data))
            cmap2 = plt.get_cmap('plasma')
            norm = Normalize(vmin=0, vmax=self.ndil - 1)
            colors = cmap2(norm(idxs % self.ndil))

            ax_qq.scatter(osm[0], osm[1], c=colors, alpha=0.7, s=12)
            sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax_qq)
            cbar.set_label(f"Index within time window (0 → {self.ndil - 1})")

            min_val = min(osm[0].min(), osm[1].min())
            max_val = max(osm[0].max(), osm[1].max())
            ax_qq.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
            ax_qq.axhline(y=-4, color='grey', linestyle='--', linewidth=1)
            ax_qq.axhline(y=4,  color='grey', linestyle='--', linewidth=1)
            ax_qq.set_xlabel("Theoretical Quantiles")
            ax_qq.set_ylabel("Experimental Quantiles")
            ax_qq.set_title("QQ-plot of transformed data")

            # =======================
            #  PP plot
            # =======================
            data_sorted = np.sort(data)
            emp_cdf = np.arange(1, len(data_sorted) + 1) / (len(data_sorted) + 1)
            th_cdf = stats.norm.cdf(data_sorted)
            idxs2 = np.arange(len(data_sorted))
            colors_pp = cmap2(norm(idxs2 % self.ndil))

            ax_pp.scatter(th_cdf, emp_cdf, c=colors_pp, alpha=0.7, s=12)
            sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
            sm2.set_array([])
            cbar2 = plt.colorbar(sm2, ax=ax_pp)
            cbar2.set_label(f"Index within time window (0 → {self.ndil - 1})")
            ax_pp.plot([0, 1], [0, 1], 'r--', lw=1)
            ax_pp.set_xlabel("Theoretical CDF")
            ax_pp.set_ylabel("Empirical CDF")
            ax_pp.set_title("PP-plot of transformed data")

            plt.tight_layout()
            plt.show()
        if figure == 'quantile functions':
            fig, ax = plt.subplots(figsize=(8, 4)) 

            cmap1 = plt.get_cmap('coolwarm')
            fq = self.spqs[site].fit_quantiles[idx * self.ndil:(idx + num_day) * self.ndil, :]
            colors1 = cmap1(np.linspace(0, 1, fq.shape[1]))

            for j in range(fq.shape[1]):
                ax.plot(fq[:, j], color=colors1[j], alpha=0.5, label=self.quantiles[j])

            xticks_pos = np.arange(0, fq.shape[0], self.ndil)
            start_date = pd.to_datetime(self.start_train) + pd.to_timedelta(idx, unit="D")
            xticks_labels = [
                (start_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(num_day)
            ]

            ax.set_xticks(xticks_pos)
            ax.set_xticklabels(xticks_labels, rotation=45)
            ax.set_title("Quantile functions")
            ax.legend()

            plt.tight_layout()
            plt.show()
        if figure == 'QQ-plot':
            if print_train:
                raw = self.quantile_train[site]
            else:
                raw = self.quantile_test[site]

            data = raw.T.flatten()

            osm, _ = stats.probplot(data, dist="norm")
            idxs = np.arange(len(data))

            cmap2 = plt.get_cmap('plasma')
            norm = Normalize(vmin=0, vmax=self.ndil - 1)
            colors = cmap2(norm(idxs % self.ndil))

            fig, ax = plt.subplots(figsize=(7, 4.5)) 

            ax.scatter(osm[0], osm[1], c=colors, alpha=0.7, s=12)

            sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(f"Index within time window (0 → {self.ndil - 1})")

            min_val = min(osm[0].min(), osm[1].min())
            max_val = max(osm[0].max(), osm[1].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
            ax.axhline(y=-4, color='grey', linestyle='--', linewidth=1)
            ax.axhline(y=4,  color='grey', linestyle='--', linewidth=1)

            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Experimental Quantiles")
            ax.set_title("QQ-plot of transformed data")

            plt.tight_layout()
            plt.show()
        if figure == 'PP-plot':
            if print_train:
                raw = self.quantile_train[site]
            else:
                raw = self.quantile_test[site]

            data = raw.T.flatten()

            data_sorted = np.sort(data)
            empirical_cdf = np.arange(1, len(data_sorted) + 1) / (len(data_sorted) + 1)

            theoretical_cdf = stats.norm.cdf(data_sorted)

            idxs = np.arange(len(data_sorted))
            cmap = plt.get_cmap('plasma')
            norm = Normalize(vmin=0, vmax=self.ndil - 1)
            colors = cmap(norm(idxs % self.ndil))

            fig, ax = plt.subplots(figsize=(7, 4.5))

            ax.scatter(theoretical_cdf, empirical_cdf, c=colors, alpha=0.7, s=12)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(f"Index within time window (0 → {self.ndil - 1})")

            ax.plot([0, 1], [0, 1], 'r--', lw=1)

            ax.set_xlabel("Theoretical CDF")
            ax.set_ylabel("Empirical CDF")
            ax.set_title("PP-plot of transformed data")

            plt.tight_layout()
            plt.show()



def save(file, obj): 
    """
    Save a OutagePipeline object to a file using joblib with gzip compression.
    
    Parameters:
    -----------
    file : str or file-like object
        The file path where the object should be saved.
    obj : OutagePipeline object
        The object to be saved.
    """
    # Ensure the directory exists
    dir_name = os.path.dirname(file)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Save the object with joblib using gzip compression
    joblib.dump(obj, file, compress=('gzip', 3))

def load(file) -> OutagePipeline:
    """
    Load a QLinear object from a file using joblib.
    
    Parameters:
    -----------
    file : str or file-like object
        The file path from which the object should be loaded.
    """
    obj = joblib.load(file)
    return obj



        

        
        