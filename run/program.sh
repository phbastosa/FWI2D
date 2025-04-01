#!/bin/bash

modeling="../src/modeling.py"
inversion="../src/inversion.py"
migration="../src/migration.py"

# Main dialogue ---------------------------------------------------------------------------------------

USER_MESSAGE="
-------------------------------------------------------------------------------
                                    \033[34mFWI\033[0;0m
-------------------------------------------------------------------------------
\nUsage:\n
    $ $0 -modeling                      
    $ $0 -inversion           
    $ $0 -migration
\nTests:\n
    $ $0 -test_modeling                      
    $ $0 -test_inversion           
    $ $0 -test_migration
    
-------------------------------------------------------------------------------
"

[ -z "$1" ] && 
{
	echo -e "\nYou didn't provide any parameter!" 
	echo -e "Type $0 -help for more info\n"
    exit 1 
}

case "$1" in

-h) 

	echo -e "$USER_MESSAGE"
	exit 0
;;

-modeling) 

    python3 -B $modeling parameters.txt
	
    exit 0
;;

-inversion) 
    
    python3 -B $inversion parameters.txt
	
    exit 0
;;

-migration) 
    
    python3 -B $migration parameters.txt
	
    exit 0
;;

-test_modeling)

    python3 -B ../tests/modeling/generate_models.py
    python3 -B ../tests/modeling/generate_geometry.py

    python3 -B $modeling ../tests/modeling/parameters.txt

    # python3 -B ../tests/modeling/generate_figures.py

	exit 0
;;

-test_inversion) 

    python3 -B ../tests/inversion/generate_models.py
    python3 -B ../tests/inversion/generate_geometry.py

    python3 -B $modeling ../tests/inversion/parameters.txt
    python3 -B $inversion ../tests/inversion/parameters.txt

    python3 -B ../tests/inversion/generate_figures.py

    exit 0
;;

-test_migration)

    python3 -B ../tests/migration/generate_models.py
    python3 -B ../tests/migration/generate_geometry.py

    python3 -B $modeling ../tests/migration/parameters.txt
    python3 -B $migration ../tests/migration/parameters.txt

    python3 -B ../tests/migration/generate_figures.py

	exit 0
;;

* ) 

	echo -e "\033[31mERRO: Option $1 unknown!\033[m"
	echo -e "\033[31mType $0 -h for help \033[m"
	
    exit 3
;;

esac