# vid generator vars
export vidFile="data/1videos/"
export vidConfigFile="cfg/vayne.yaml"
export vidPrefix="test_"

# crop vars
export cropFile="data/2cropped_images/"
export cropPrefix="test"
export skipFrames=40

# bootstrap vars
export datasetFile="data/3data_set/"
export countFile="count.yaml"
export mapFile="data/0map/"
export datasetSize=11
export labelFile="cfg/vayne.names"

# split vars
export datasetFinal="/home/hunter/Games/vayneDataset/"
export trainingSetSize=8
export datasetCfg="/home/hunter/Games/cfg/"
export datasetName="vayneTest"
export top=2

#python3 scripts/video_generator.py -c "$vidConfigFile" -o "$vidFile" -p "$vidPrefix"

# Remove background, create images of sprite only
#python3 scripts/frameExporter.py -o "$cropFile" -c "$vidConfigFile" -p "$cropPrefix" -i "$vidFile" -s "$skipFrames"  -q "$vidPrefix"

# Generate the data set
#python3 scripts/bootstrap.py -c "$vidConfigFile" -o "$datasetFile" -k "$countFile" -i "$cropFile" -q "$cropPrefix" -m "$mapFile" -n "$datasetSize" -l "$labelFile"

# organize the dataset for training
python3 scripts/split_train_test_darknet_style.py -i "$datasetFile" -c "$vidConfigFile" -o "$datasetFinal" -s "$trainingSetSize" -k "$datasetCfg" -n "$datasetName" -t "$top"
