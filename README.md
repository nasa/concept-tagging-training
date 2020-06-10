# Concept Tagging Training

This software enables the creation of concept classifiers, to be utilized by an 
accompanying [service](https://developer.nasa.gov/DataSquad/sti-service-compose). 

You can see a list of options for this project by navigating to the root of the project and executing `make` or `make help`.

This project requires:
* [docker](https://docs.docker.com/install/) -- [tested with this version](docker-versions.txt)
* [GNU Make](https://www.gnu.org/software/make/) -- tested with 3.81 built for i386-apple-darwin11.3.0

## Index:
1. [installation](#installation)
2. [how to run](#how-to-run)
3. [managing experiments](#managing-experiments)
4. [advanced usage](#advanced-usage)

## installation
You have several options for installing and using the pipeline. 
1) [pull existing docker image](#pull-existing-docker-image)
2) [build docker image from source](#build-docker-image-from-source)
3) [install in python virtual environment](#install-in-python-virtual-environment)
 
### pull existing docker image
You can just pull a stable docker image which has already been made:
```bash
docker pull storage.analytics.nasa.gov/abuonomo/concept_trainer:stable
```
In order to do this, you must be on the NASA network and able to connect to the <https://storage.analytics.nasa.gov> docker registry.
\* <sub> There are several versions of the images. You can see them [here](https://storage.analytics.nasa.gov/repository/abuonomo/rat_trainer). 
If you don't use "stable", some or all of this guide may not work properly. </sub>


### build docker image from source
To build from source, first clone this repository and go to its root.

Then build the docker image using:
```bash
docker build -t concept_trainer:example .
```
Substitute `concept_trainer:example` for whatever name you would like. Keep this image name in mind. It will be used elsewhere. 

\* If you are actively developing this project, you should look at the `make build` in [Makefile](Makefile). This command automatically tags the image with the current commit url and most recent git tag. The command requires that [setuptools-scm](https://pypi.org/project/setuptools-scm/) is installed.

### install in python virtual environment
\* tested with python3.7
First, clone this repository. 
Then create and activate a virtual environment. For example, using [venv](https://docs.python.org/3/library/venv.html):
```bash
python -m venv my_env
source my_env/bin/activate
```
Next, while in the root of this project, run `make requirements`.


## how to run
The pipeline takes input document metadata structured like [this](data/raw/STI_public_metadata_records_sample100.jsonl) and a config file like [this](config/test_config.yml). The pipeline produces interim data, models, and reports.

1. [using docker](#using-docker) -- if you pulled or built the image
2. [using python in virtual environment](#using-python-in-virtual-environment) -- if you are running in a local virtual environment

### using docker
First, make sure `config`, `data`, `data/raw`, `data/interim`, `models`, and `reports` directories. If they do not exist, make them (`mkdir config data models reports data/raw`). These directories will be used as docker mounted volumes. If you don't make these directories beforehand, they will be created by docker later on, but their permissions will be unnecessarily restrictive.  

Next, make sure you have your input data in the `data/raw/` directory. [Here](data/raw/STI_public_metadata_records_sample100.jsonl) is an example file with the proper structure. You also need to make sure the `subj_mapping.json` file [here](data/interim/subj_mapping.json) is in `data/interim/` directory.

Now, make sure you have a config file in the `config` directory. [Here](config/test_config.yml) is an example config which will work with the above example file.

With these files in place, you can run the full pipeline with this command:
```bash
docker run -it \
     -v $(pwd)/data:/home/data \
     -v $(pwd)/models:/home/models \
     -v $(pwd)/config:/home/config \
     -v $(pwd)/reports:/home/reports \
    concept_trainer:example pipeline \
        EXPERIMENT_NAME=my_test_experiment \
        IN_CORPUS=data/raw/STI_public_metadata_records_sample100.jsonl \
        IN_CONFIG=config/test_config.yml
```
Substitute `concept_trainer:example` with the name of your docker image.
You can set the `EXPERIMENT_NAME` to whatever you prefer.
`IN_CORPUS` and `IN_CONFIG` should be set to the paths to the corpus and to the configuration file, respectively.

\* Developers can also use the `container` command in the [Makefile](Makefile). Note that this command requires [setuptools-scm](https://pypi.org/project/setuptools-scm/). Note that this command will use the image defined by the `IMAGE_NAME` variable and version number equivalent to the most recent git tag. 


### using python in virtual environment

Assuming you have cloned this repository, files for testing the pipeline should be in place. In particular, `data/raw/STI_public_metadata_records_sample100.jsonl` and `config/test_config.yml` should both exist. Additionally, you should add the `src` directory to your `PYTHONPATH`:
```
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/
``` 

Then, you can run a test of the pipeline with: 
```
make pipeline \
    EXPERIMENT_NAME=test \
    IN_CORPUS=data/raw/STI_public_metadata_records_sample100.jsonl \
    IN_CONFIG=config/test_config.yml
```
If you are not using the default values, simply substitute the proper paths for `IN_CORPUS` and `IN_CONFIG`. Choose whatever name you prefer for `EXPERIMENT_NAME`.

## managing experiments

If you have access to the `hq-ocio-ci-bigdata` moderate s3 bucket, you can sync local experiments with those in the s3 bucket.

For example, if you created a local experiment with `EXPERIMENT_NAME=my_cool_experiment`, you can upload your local results to the appropriate place on the s3 bucket with:
```bash
make sync_experiment_to_s3 EXPERIMENT_NAME=my_cool_experiment PROFILE=my_aws_profile
```
where `my_aws_profile` is the name of your awscli profile which has access to the given bucket.

Afterwards, you can download the experiment interim files and results with:
```bash
make sync_experiment_from_s3 EXPERIMENT_NAME=my_cool_experiment PROFILE=my_aws_profile
```
## use full sti metadata records
If you have access to the moderate bucket and you want to work with the full STI metadata records, you can download them to the `data/raw` folder with:
```bash
make sync_raw_data_from_s3 PROFILE=my_aws_profile
``` 
When using these data, you will want to use a config file which is different from the test config file. You can browse previous experiments at `s3://hq-ocio-ci-bigdata/home/DataSquad/classifier_scripts/` to see example config files. You might try:
```yaml
weights:  # assign weights for term types specified in process section
  NOUN: 1
  PROPN: 1
  NOUN_CHUNK: 1
  ENT: 1
  ACRONYM: 1
min_feature_occurrence: 100
max_feature_occurrence: 0.6
min_concept_occurrence: 500
```
See [config/test_config.yml](config/test_config.yml) for details on these parameters.

## advanced usage
For more advanced usage of the project, look at the [Makefile](Makefile) commands and their associated scripts. You can learn more about these python scripts by them with help flags. For example, you can run `python src/make_cat_models.py -h`. 

