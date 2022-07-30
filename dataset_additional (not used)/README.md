# New Datasets (Not used in current model yet)

Newly collected dataset on 2022-4-4. This contains the data we intend to replace the old data with. But this folder isn't yet fully merged with the TFT models. However they are fully compatible and ready to merge. We can do the replacement after stabilizing the models.

## Steps to Reproduce

Download the raw dataset folder from [Google Drive](https://drive.google.com/drive/folders/1mnvSJx9kKc3E3pcsDDplaGZDG1pndLEl?usp=sharing). Unzip and copy the **raw** folder in this current repository. Then run the data cleaning notebook. The cleaned csv files will be written in the **cleaned** folder. The dynamic features are pivoted and data before date 2020-03-01 are dropped. County '*FIPS*' code is used as the primary key across different files.

`Rurality_Median_Mad` and `RuralityCodes` have been copied from the `dataset_raw` folder.

## Dataset Description

Each of the following features are collected at county level. Source links are added for the raw dataset. Which needs to be processed to get the clean dataset here. Note that, the `config.json` may use only a subset of the features listed here.

* `Cases`: Daily covid cases by County. This file is preprocessed from cumulative covid cases collected from [USAFacts](https://usafacts.org/visualizations/coronavirus-covid-19-spread-map). Download from [here](https://static.usafacts.org/public/data/covid-19/covid_confirmed_usafacts.csv?_ga=2.171350432.1090772774.1649722578-318392369.1648433378).
* `Deaths`: Daily covid deaths by County. This file is preprocessed from cumulative covid deaths collected from [USAFacts](https://usafacts.org/visualizations/coronavirus-covid-19-spread-map). Download from [here](https://static.usafacts.org/public/data/covid-19/covid_deaths_usafacts.csv?_ga=2.116740742.1090772774.1649722578-318392369.1648433378).
* `Testing`: From [US Covid Atlas](https://theuscovidatlas.org/). Download the [County - Testing Counts - CDC](https://theuscovidatlas.org/download) file.
* `Health rank measure`: Health measures and rankings data from CHR updated in 2021. Link to [data](https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data2021.csv) and [data dictionary](https://www.countyhealthrankings.org/sites/default/files/media/document/DataDictionary_2021.pdf). The cleaned data has the following columns. The comorbidity columns are the percent of the population belonging to that comorbidity.
  * % Fair or Poor Health
  * % Smokers
  * % Adults with Obesity
  * Primary Care Physicians Rate
  * % Flu Vaccinated
  * Average Daily PM2.5
  * % Severe Housing Problems
* `Mobility bts`: [Trips by distance data](https://data.bts.gov/Research-and-Statistics/Trips-by-Distance/w96p-f2qv) calculated by The Bureau of Transportation Statistics (BTS) of U.S.
  * Number of Trips
  * Population Not Staying at Home
* `Mobility google`: POI(Points of Interest) based mobility for '*workplaces_percent_change_from_baseline*' from [Google community report](https://www.google.com/covid19/mobility/). The raw dataset has more mobility columns, but we have dropped them due to many missing values.
* `Population`: Population demographics collected from [US govt census data](https://www.statsamerica.org/downloads/default.aspx) from 2019. The population density is calculated by dividing the '*Total Population*' with '*Area of Land (square kilometers)*' from the [Average_Household_Size_and_Population_Density_-_County](https://covid19.census.gov/datasets/21843f238cbb46b08615fc53e19e0daf_1/explore?showTable=true) data. Our cleaned data contains the following columns,
  * Total Population
  * Population 0-4
  * Population 5-17
  * Population 18-24
  * Population 25-44
  * Population 45-64
  * Population 65+
  * Population Under 18
  * Population 18-54
  * Population 55+
  * Male Population
  * Female Population
  * Population Density (people per square kilometer)
* `Rurality_Median_Mad`: Contains the median and mad value calculated from RuralityCodes.
* `RuralityCodes`: Contains the rurarility ranks counted for each county.
* `Social vulnerability index`: Social Vulnerability Index (SVI) indicates the relative vulnerability of every U.S. county. [This dataset](https://coronavirus-resources.esri.com/datasets/cdcarcgis::overall-svi-counties/about) visualizes the 2018 overall SVI for U.S. counties. Raw dataset has 15 social factors grouped into four major themes. We have only kept the four major themes, and the overall rank in the cleaned version. For a detailed description of variable uses, please refer to the [full SVI 2018 documentation](https://svi.cdc.gov/Documents/Data/2018_SVI_Data/SVI2018Documentation.pdf).
  * *RPL_THEME1*: Socioeconomic
  * *RPL_THEME2*: Housing Composition and Disability
  * *RPL_THEME3*: Minority Status and Language
  * *RPL_THEME4*: Housing and Transportation
  * *RPL_THEMES*: Overall rank
* `Vaccination`: Overall US COVID-19 Vaccine administration and vaccine equity data at [county level from CDC](https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh). The earliest available entry is from 2020-12-13. The raw data contains 66 columns in total, but we are using the following features,
  * `Administered_Dose1_Recip`: Total number of people with at least one dose of vaccine.
  * `Series_Complete_Yes`: Total number of people who are fully vaccinated (have a second dose of a two-dose vaccine or one dose of a single-dose vaccine) based on the jurisdiction and county where recipient lives.

## Status on Old Dataset

Most of them are not linked to any verified source or I couldn't find one. Hence suggesting the replacement of some of them.

* `2020votes`: not used and don’t need to be updated.
* `Age Distribution`: Source not found. Used population by age instead.
* `Air pollution`: Source not found. Used Pm2.5 air pollution index instead.
* `Alaskavoting2016`: not used and no need to update.
* `Comorbidities`: Source not found. Used other comorbidities from US health ranking instead.
* `Demographic`: Which demographic? Used demographic data from US population census instead (population by age, sex).
* `Disease Spread`: Source not found. How is it calculated?
* `Health Disparities`: Doesn’t mention source. But we can use US health ranking instead.
* `Hospital beds`: Most data are either noise, lots of missing values or start at the later part of 2020.
* `Mobility`: Replaced with google, apple and bts data.
* `Population`: Demographic data added.
* `Residential density`: Didn’t find this in US census data. Can be replaced with population density.
* `Rurality_Median_Mad`: can be calculated from rurality score.
* `RuralityCodes`: Sources found for 2013. But it doesn't match.
* `Social Distancing`: Private source not free anymore. Is there any other source?
* `Testing`: updated.
* `Transmissible Cases`: Source not found.
* `US_daily_cumulative_cases`: Updated.
* `US_daily_cumulative_deaths`: Updated.
* `Vaccination`: Updated
* `VaccinationOneDose`: Updated
  