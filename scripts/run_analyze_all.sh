#!/bin/sh
    
python src/analyze.py experiment.atlas.center=hafenLowredshiftLymanLimit2017 experiment.cartography.required_pub_conditions.fields_of_study="[Physics]"

python src/analyze.py experiment.atlas.center=Imeletal2022 experiment.cartography.required_pub_conditions.fields_of_study="[Linguistics]"

python src/analyze.py experiment.atlas.center=Ololube2012 experiment.cartography.required_pub_conditions.fields_of_study="[Education]"

python src/analyze.py experiment.atlas.center=Torres2013 experiment.cartography.required_pub_conditions.fields_of_study="[Medicine]"

python src/analyze.py experiment.atlas.center=West2003 experiment.cartography.required_pub_conditions.fields_of_study="[Economics]"

python src/analyze.py experiment.atlas.center=Bacon2019 experiment.cartography.required_pub_conditions.fields_of_study="[Philosophy]"

python src/analyze.py experiment.atlas.center=Miele2022 experiment.cartography.required_pub_conditions.fields_of_study="[Materials Science]"

python src/analyze.py experiment.atlas.center=ForeroOrtega2021 experiment.cartography.required_pub_conditions.fields_of_study="[Geology]"

python src/analyze.py experiment.atlas.center=Andre2018 experiment.cartography.required_pub_conditions.fields_of_study="[Mathematics]"

python src/combine_all_data.py analysis_data/all_data.csv
