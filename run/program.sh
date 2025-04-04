#!/bin/bash

admin="../src/admin/admin.cpp"

geometry="../src/geometry/geometry.cpp"

modeling="../src/modeling/modeling.cu"
modeling_main="../src/modeling_main.cpp"

inversion="../src/inversion/inversion.cu"
inversion_main="../src/inversion_main.cpp"

migration="../src/migration/migration.cu"
migration_main="../src/migration_main.cpp"

flags="-Xcompiler -fopenmp --std=c++11 -lm -lfftw3 -O3"

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

-compile) 

    echo -e "Compiling stand-alone executables!\n"

    echo -e "../bin/\033[31mmodeling.exe\033[m" 
    nvcc $admin $geometry $modeling $modeling_main $flags -o ../bin/modeling.exe

    # echo -e "../bin/\033[31minversion.exe\033[m" 
    # nvcc $admin $geometry $modeling $inversion $inversion_main $flags -o ../bin/inversion.exe

    echo -e "../bin/\033[31mmigration.exe\033[m"
    nvcc $admin $geometry $modeling $migration $migration_main $flags -o ../bin/migration.exe

    exit 0
;;

-modeling) 

    ./../bin/modeling.exe parameters.txt
	
    exit 0
;;

-inversion) 
    
    ./../bin/inversion.exe parameters.txt
	
    exit 0
;;

-migration) 
    
    ./../bin/migration.exe parameters.txt
	
    exit 0
;;

-test_modeling)

    python3 -B ../tests/modeling/generate_models.py
    python3 -B ../tests/modeling/generate_geometry.py

    ./../bin/modeling.exe ../tests/modeling/parameters.txt

    python3 -B ../tests/modeling/generate_figures.py ../tests/modeling/parameters.txt

	exit 0
;;

-test_inversion) 

    # python3 -B ../tests/inversion/generate_models.py
    # python3 -B ../tests/inversion/generate_geometry.py

    # python3 -B $modeling ../tests/inversion/parameters.txt
    # python3 -B $inversion ../tests/inversion/parameters.txt

    # python3 -B ../tests/inversion/generate_figures.py

    exit 0
;;

-test_migration)

    python3 -B ../tests/migration/generate_models.py
    python3 -B ../tests/migration/generate_geometry.py

    # ./../bin/modeling.exe ../tests/migration/parameters.txt
    ./../bin/migration.exe ../tests/migration/parameters.txt

    # python3 -B ../tests/migration/generate_figures.py ../tests/migration/parameters.txt

	exit 0
;;

* ) 

	echo -e "\033[31mERRO: Option $1 unknown!\033[m"
	echo -e "\033[31mType $0 -h for help \033[m"
	
    exit 3
;;

esac