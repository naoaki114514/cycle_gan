#!/bin/bash
#sh dataset.sh file_name
FILE=$1 #コマンドライン引数 ダウンロードするファイル名指定

if [ -z "$FILE" ]; then
   echo "please write dataset_name on command line"
   exit 1
fi

if [[ $FILE != "ae_photos" && $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebra" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" && $FILE != "maps" && $FILE != "cityscapes" && $FILE != "facades" && $FILE != "iphone2dslr_flower" && $FILE != "ae_photos" ]]; then
    echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
    exit 1
fi

URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE=./$FILE.zip
TARGET=./$FILE
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d .
rm $ZIP_FILE

mkdir ./data
mkdir ./data/train
mkdir ./data/test
mkdir ./data/val

mv "$TARGET/trainA" "./data/train/A"
mv "$TARGET/trainB" "./data/train/B"
mv "$TARGET/testA" "./data/test/A"
mv "$TARGET/testB" "./data/test/B"
mv "$TARGET/valA" "./data/val/A"
mv "$TARGET/valB" "./data/val/B"

rm -rf $FILE