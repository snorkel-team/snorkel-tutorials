import logging

import dask.dataframe as dd
from snorkel.labeling import LabelModel
from snorkel.labeling.apply.dask import DaskLFApplier

from drybell_lfs import article_mentions_person, body_contains_fortune, person_in_db

logging.basicConfig(level=logging.INFO)


def main(data_path, output_path):
    # Read data
    logging.info(f"Reading data from {data_path}")
    data = dd.read_parquet(data_path)
    data = data.repartition(npartitions=2)

    # Build label matrix
    logging.info("Applying LFs")
    lfs = [article_mentions_person, body_contains_fortune, person_in_db]
    applier = DaskLFApplier(lfs)
    L = applier.apply(data)

    # Train label model
    logging.info("Training label model")
    label_model = LabelModel(cardinality=2)
    label_model.fit(L)

    # Generate training labels
    logging.info("Generating probabilistic labels")
    y_prob = label_model.predict_proba(L)[:, 1]
    data = data.reset_index().set_index("index")
    data_labeled = data.assign(y_prob=dd.from_array(y_prob))
    dd.to_parquet(data_labeled, output_path)
    logging.info(f"Labels saved to {output_path}")


if __name__ == "__main__":
    main("drybell/data/raw_data.parquet", "drybell/data/labeled_data.parquet")
