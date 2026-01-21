# Solar Data Tools Webinar

*Presented 25 September 2025*

This directory contain the marimo notebook and presentation given at the Solar Data Tools webinar, hosted by NREL on 9/25/25.

## Instructions

Starting in a fresh Python 3.12 virtual environment, install the requirements file in this directory (`solar-data-tools/webinar/requirements.txt`). From this directory, run:

```
pip install -r requirements.txt
```

Then, the marimo notebook can be run in two ways. To see the code and try out new things, run:

```
marimo edit SDT_demo.py
```

To view the presentation without the code blocks, run:


```
marimo run SDT_demo.py
```

## Warning

In the live webinar, I use MOSEK to speed up the statistical clear sky fitting module. For this publicly hosted version, we're assuming people don't have MOSEK, so this has been turned off. So, that step will take ~8 minutes to complete, rather than the ~45 seconds seen in the live demo.
