if [ -z "$2" ]
then
  a=500000
  b=1000000
  python jet_substructure.py --start_row 0 --stop_row $a --file_name $1 &
  python jet_substructure.py --start_row $a --stop_row $b --file_name $1 &
  wait
  c='clustered_'$1
  python combine.py --file_name $c
else
  a=550000
  b=1100000
  python jet_substructure.py --start_row 0 --stop_row $a --has_labels --file_name $1 &
  python jet_substructure.py --start_row $a --stop_row $b --has_labels --file_name $1 &
  wait
  c='clustered_'$1
  python combine.py --has_labels --file_name $c
fi
