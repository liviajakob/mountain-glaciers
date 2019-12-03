
#Execution example: ./demDiffMad.sh hma-gridcells.txt
#Execution example: .demDiffMad.sh alaska-gridcells.txt
#Execution example with time display: time ./demDiffMad.sh hma-gridcells.txt

#Sets input variable to be the first variable. This should be the command file.
export input=$1

#Extract columns from command file to variables
minX=($(awk -F"," '{print $1}' $input))
maxX=($(awk -F"," '{print $2}' $input))
minY=($(awk -F"," '{print $3}' $input))
maxY=($(awk -F"," '{print $4}' $input))

#Run in parallel passing in variables row by row
# -j number of parallel processes
#-k parameter enforces the processing to be in the same order as input
parallel --progress --link -j 2 'python CmdDemDiffMad.py {1} {2} {3} {4}' ::: ${minX[*]} ::: ${maxX[*]} ::: ${minY[*]} ::: ${maxY[*]}
