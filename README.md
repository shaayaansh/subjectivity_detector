# subjectivity_detector

We have three different datastets labeled for subjectivity classification. MPQA, News-1 and News-2.

We train a baseline model in different settings on these datasets:
* Multi-Dataset Learning: where we train and test in cross-validation setting.
* Multi-task Learning: where we view each dataset as a different task and allocate it's own classification head to it. 
