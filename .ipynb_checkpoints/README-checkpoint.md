# training_income_causal_inference
Using a real life dataset the impact of job training program on participant's income is evaluated.

# About Dataset
Famous dataset to estimate the impact of a training program (National Supported Work Demonstration) on the income of beneficiaries in 1978.
Most of variables are selfexplanatory except the below ones.

treat = Treatment indicator: if the individual participated in the job training program, 0 if not (control group).

re74 = Real earnings in 1974 (pre-treatment): Income from a year before the program started.

re75 = Real earnings in 1975 (pre-treatment): Another pre-treatment income measure, useful to control for prior economic status.

re78 = Real earnings in 1978 (post-treatment): The outcome variable. It measures how much a person earned after the job training, used to assess the program’s effect.

# Package content
This package includes a nootbok (in dev folder), the dataset in data folder, and the callable python code in src folder.
