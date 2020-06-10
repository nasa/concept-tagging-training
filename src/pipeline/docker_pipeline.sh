#!/usr/bin/env bash

#export IMAGE=storage.analytics.nasa.gov/abuonomo/rat_trainer:dev # TODO: make command line arg
## TODO: add help for description of parameters
#
#echo "Reading from ${IN_DATA}." # TODO: put in docker container env variable so its in docker logs
#echo "Dumping to ${OUT_DATA}."

usage="$(basename "$0") [-h] [-i path] [-o path] [-d docker-image] [-l loglevel] [-c cpus]
Concept training pipeline

where:
    -h  show this help text
    -i  (absolute path) input data directory
    -o  (absolute path) output data directory
    -d  the docker image to use
    -l  the log level to use ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    -c  number of cpus to allow container to use"

# Get command line arguments
input=""
output=""
while getopts ':hi:o:d:l:c:' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    i) input=$OPTARG
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
    o) output=$OPTARG
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
    d) image=$OPTARG
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
    l) LOGLEVEL=$OPTARG
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
    c) cpus=$OPTARG
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
    \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

# Check for errors
if [ ! -d "${input}" ]; then
    echo "${input} directory does not exists. Choose a directory name which does exists and contains requisite data."
    exit 1
fi
if [ -d "${output}" ]; then
    echo "${output} directory already exists. Choose a new directory name which does not exist."
    exit 1
fi
if [ "${LOGLEVEL}" = "" ]; then
   echo "Setting empty LOGLEVEL to INFO."
   export LOGLEVEL="INFO"
fi
if [ "${cpus}" = "" ]; then
   echo "Setting empty LOGLEVEL to INFO."
   export cpus=0.000
fi


mkdir ${output}

echo "Running full pipeline."
docker run -it\
    -v ${input}:/home/pipeline/volumes/in_data \
    -v ${output}:/home/pipeline/volumes/out_data \
    -e LOGLEVEL=${LOGLEVEL} \
    --cpus=${cpus} \
    ${image} bash -c 'bash pipeline/start.sh'
#--cpus=<value> # TODO: add cpus arg
echo "Completed Pipeline."
