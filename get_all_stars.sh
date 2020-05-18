python colors.py -l > types.txt

while read line
do 
	echo $line
    	python colors.py --action="save_star" --star_type=$line
done < types.txt
rm types.txt

