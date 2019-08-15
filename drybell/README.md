# Snorkel Drybell Example

This example is based on the
[Snorkel Drybell project](https://ai.googleblog.com/2019/03/harnessing-organizational-knowledge-for.html),
a collaboration between the Snorkel team and Google to implement weak supervision at industrial scale.
You can read more in the
[blog post](https://ai.googleblog.com/2019/03/harnessing-organizational-knowledge-for.html)
and [research paper (SIGMOD Industry, 2019)](https://arxiv.org/abs/1812.00417).
The paper used a running example of classifying documents as containing a celebrity mention or not,
which is what we use here as well.
The data is a very small set of six faux newspaper articles and titles, stored as a
[Parquet file](https://parquet.apache.org/):

```
Title                                       Body
-----                                       ----
Sports team wins the game!                  It was an exciting game. The team won at the end.
Jennifer Smith donates entire fortune.      She has a lot of money. Now she has less, because...
...
```

Of course, with such a small (and very fake) dataset, we don't expect to produce
high quality models.
The goal here is to demonstrate how Snorkel can be used in a large-scale production setting.
We present two scripts —
one using Snorkel's [Dask](https://dask.org/) interface
and one using Snorkel's [Spark](https://spark.apache.org/) interface
— that represent how Snorkel can be deployed as part of a pipeline.
We also demonstrate Snorkel's `NLPLabelingFunction` interface, similar to the 
`NLPLabelingFunction` template presented in the Drybell paper.

If you plan to execute these scripts, do so from the `snorkel-tutorials` directory:

```bash
python3 drybell/drybell_dask.py

# or

python3 drybell/drybell_spark.py
```
