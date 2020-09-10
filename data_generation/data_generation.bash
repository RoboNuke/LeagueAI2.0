# vid generator vars
export vidFile="data/1videos/vayne/"
export vidConfigFile="cfg/vayne.yaml"
export vidPrefix="vayne_"

# crop vars
export cropFile="data/2cropped_images/"
export cropPrefix="vayne_"
export skipFrames=5

# bootstrap vars
export datasetFile="data/3data_set/vayne"
export countFile="count.yaml"
export mapFile="data/0map/"
export datasetSize=6000
export labelFile="cfg/vayne.names"
export width="960"
export height="540"

# split vars
export datasetFinal="data/vayneDataset/"
export trainingSetSize=1000
export datasetCfg="cfg/"
export datasetName="vayne"
export top=2

#python3 scripts/video_generator.py -c "$vidConfigFile" -o "$vidFile" -p "$vidPrefix"

# Remove background, create images of sprite only
#python3 scripts/frameExporter.py -o "$cropFile" -c "$vidConfigFile" -p "$cropPrefix" -i "$vidFile" -s "$skipFrames"  -q "$vidPrefix"

# Generate the data set
python3 scripts/bootstrap.py -c "$vidConfigFile" -o "$datasetFile" -k "$countFile" -i "$cropFile" -q "$cropPrefix" -m "$mapFile" -n "$datasetSize" -l "$labelFile" -w "$width" -j "$height"

# organize the dataset for training
python3 scripts/split_train_test_darknet_style.py -i "$datasetFile" -c "$vidConfigFile" -o "$datasetFinal" -s "$trainingSetSize" -k "$datasetCfg" -n "$datasetName" -t "$top"
