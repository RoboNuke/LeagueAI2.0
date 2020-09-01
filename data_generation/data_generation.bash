# vid generator vars
export vidFile="data/1videos/"
export vidConfigFile="cfg/vayne.yaml"
export vidPrefix="test_"

# crop vars
export cropFile="data/2cropped_images/"
export cropPrefix="test"
export skipFrames=40

#python scripts/video_generator.py -c "$vidConfigFile" -o "$vidFile" -p "$vidPrefix"


# Remove background, create images of sprite only
python scripts/frameExporter.py -o "$cropFile" -c "$vidConfigFile" -p "$cropPrefix" -i "$vidFile" -s "$skipFrames"  -q "$vidPrefix"

# Generate the data set
#bootstrap.py

# organize the dataset for training
#split_test_train.py
