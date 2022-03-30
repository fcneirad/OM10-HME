echo 'Decompressing data for events'

for file in CDFs/all_events/*.gz
do
	gunzip $file
done

for file in CDFs/caustic_crossings/strong/*.gz
do
	gunzip $file
done

for file in CDFs/caustic_crossings/weak/*.gz
do
	gunzip $file
done

for file in CDFs/caustic_crossings/single/*.gz
do
	gunzip $file
done

echo 'Done'
