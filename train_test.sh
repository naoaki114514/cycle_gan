#!/bin/bash
#sh train_test.sh save_result_folder_name cuda_number
FOLDER=$1 #コマンドライン引数 
CUDA_NUMBER=$2

if [ -z "$FOLDER" ]; then
   echo "please write save_result_folder_name on command line"
   exit 1
fi

if [ -z "$CUDA_NUMBER" ]; then
   echo "please write cuda_number on command line"
   exit 1
fi

mkdir ./result
mkdir ./result/$FOLDER #./resultの下に結果を保存するフォルダを作成

export CUDA_VISIBLE_DEVICES=$CUDA_NUMBER #使用するGPU番号
python experiment.py $FOLDER #trainする
python test.py $FOLDER #全てのエポックでtrainされたnetworkにtest画像を全て試す