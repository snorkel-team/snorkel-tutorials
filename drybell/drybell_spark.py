import logging

import pyspark.sql.functions as F
from pyspark import SparkContext
from pyspark.sql import SQLContext
from snorkel.labeling import LabelModel
from snorkel.labeling.apply.spark import SparkLFApplier

from drybell_lfs_spark import (
    article_mentions_person,
    body_contains_fortune,
    person_in_db,
)

logging.basicConfig(level=logging.INFO)


def main(data_path, output_path):
    # Read data
    logging.info(f"Reading data from {data_path}")
    sc = SparkContext()
    sql = SQLContext(sc)
    data = sql.read.parquet(data_path)

    # Build label matrix
    logging.info("Applying LFs")
    lfs = [article_mentions_person, body_contains_fortune, person_in_db]
    applier = SparkLFApplier(lfs)
    L = applier.apply(data.rdd)

    # Train label model
    logging.info("Training label model")
    label_model = LabelModel(cardinality=2)
    label_model.fit(L)

    # Generate training labels
    logging.info("Generating probabilistic labels")
    y_prob = label_model.predict_proba(L)[:, 1]
    y_prob_sql_array = F.array([F.lit(y) for y in y_prob])
    data_labeled = data.withColumn("y_prob", y_prob_sql_array)
    data_labeled.write.mode("overwrite").parquet(output_path)
    logging.info(f"Labels saved to {output_path}")


if __name__ == "__main__":
    main("drybell/data/raw_data.parquet", "drybell/data/labeled_data_spark.parquet")
