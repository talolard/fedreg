#!/bin/bash
mkdir -p ./xmls
for YEAR in {2000..2020}
do
  TMPFILE=`mktemp`
  PWD=`pwd`
  wget https://www.govinfo.gov/bulkdata/FR/$YEAR/FR-$YEAR.zip -O $TMPFILE
  unzip -d $YEAR $TMPFILE
  rm $TMPFILE
  mv ./$YEAR/*/*.xml  -t ./xmls
done

find ./xmls -maxdepth 1 -type f -print0 | xargs -0 -n 1 -P 3 python ./convert_to_json.py
