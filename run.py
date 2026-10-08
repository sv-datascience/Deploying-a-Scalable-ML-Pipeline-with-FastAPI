#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# DO NOT MODIFY
def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    
    run = wandb.init(project="nyc_airbnb", group="cleaning", save_code=True)
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)
    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Step 6: TODO
    # Only implement this step when reaching Step 6: Pipeline Release and Updates
    # in the project.
    # Add longitude and latitude filter to allow test_proper_boundaries to pass
    # ENTER CODE HERE
    # >>> STEP 6: remove the "# " from the two lines below ONLY after the v1.0.0 run on sample2.csv fails <<<
    # idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    # df = df[idx].copy()

    # Save the cleaned data
    df.to_csv('clean_sample.csv',index=False)

    # log the new data.
    artifact = wandb.Artifact(
     args.output_artifact,
     type=args.output_type,
     description=args.output_description,
 )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


# TODO: In the code below, fill in the data type for each argument. The data type should be str, float or int. 
# TODO: In the code below, fill in a description for each argument. The description should be a string.
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")
  
    parser.add_argument(
        "--input_artifact", 
        type = str,
        help = "Fully-qualified name of the input artifact (raw data) to clean, e.g. sample.csv:latest",
        required = True
    )

    parser.add_argument(
        "--output_artifact", 
        type = str,
        help = "Name for the output artifact containing the cleaned data, e.g. clean_sample.csv",
        required = True
    )

    parser.add_argument(
        "--output_type", 
        type = str,
        help = "Type of the output artifact, e.g. clean_sample",
        required = True
    )

    parser.add_argument(
        "--output_description", 
        type = str,
        help = "A short description of the output artifact",
        required = True
    )

    parser.add_argument(
        "--min_price", 
        type = float,
        help = "Minimum rental price to keep; listings below this price are dropped as outliers",
        required = True
    )

    parser.add_argument(
        "--max_price",
        type = float,
        help = "Maximum rental price to keep; listings above this price are dropped as outliers",
        required = True
    )


    args = parser.parse_args()

    go(args)
