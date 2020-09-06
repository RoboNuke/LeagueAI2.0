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
export datasetSize=10
export labelFile="cfg/vayne.names"
#python3 scripts/video_generator.py -c "$vidConfigFile" -o "$vidFile" -p "$vidPrefix"


# Remove background, create images of sprite only
#python3 scripts/frameExporter.py -o "$cropFile" -c "$vidConfigFile" -p "$cropPrefix" -i "$vidFile" -s "$skipFrames"  -q "$vidPrefix"

# Generate the data set
python3 scripts/bootstrap.py -c "$vidConfigFile" -o "$datasetFile" -k "$countFile" -i "$cropFile" -q "$cropPrefix" -m "$mapFile" -n "$datasetSize" -l "$labelFile"

# organize the dataset for training
#split_test_train.py
