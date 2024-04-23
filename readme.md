# InfiniTune

Welcome to InfiniTune, a groundbreaking tool developed by students at GIKI, designed to automatically uncover optimal configurations for a DBMS's knobs. InfiniTune's unique feature includes a feedback loop mechanism, enabling the model to continuously learn and enhance its tuning capabilities as it interacts with databases. This not only streamlines the knob tuning process of a DBMS for individuals lacking specialized knowledge in database administration but also ensures that InfiniTune becomes progressively more proficient over time, refining its tuning process with each iteration.

## MacOS Quick Setup

### Prerequisites

- *Platform:* MacOS 13.2.1+
- *Python3:* Ensure Python3 is installed on your system. If not, you can install it via Homebrew:
  
  bash
  brew update
  brew install python3
  

### Installation Steps

1. *Clone InfiniTune Repository:*

   bash
   git clone git@github.com:WardahTariq/InfiniTune.git
   

2. *Install PostgreSQL:*

   bash
   brew install postgresql
   

3. *Install Required Packages:*

   Navigate to the cloned InfiniTune repository directory and install the required Python packages:
   
   bash
   cd InfiniTune
   pip3 install -r requirements.txt
   

## Getting Started

1. *Access PostgreSQL Configuration File Location:*

   Open a terminal and enter the PostgreSQL command-line interface:

   bash
   psql postgres
   

   Within the PostgreSQL command-line interface, run the following command to show the location of the configuration file:

   sql
   SHOW config_file;
   

   Copy the location of the PostgreSQL configuration file displayed.

2. *Configure InfiniTune:*

   Open the config.py file in the cloned InfiniTune repository directory and paste the copied PostgreSQL configuration file path in the conf_path variable:

   python
   conf_path = 'your_copied_configuration_file_path'
   

3. *Run InfiniTune:*

   Execute the Evaluation.py script:

   bash
   python3 Evaluation.py
   

   You will be prompted to enter your database name and related details such as threads and clients. After entering the required information, the model will tune your database, and the final knob configurations will be installed in your PostgreSQL.

Now you're all set to optimize your DBMS configurations effortlessly with InfiniTune! If you encounter any issues or have feedback, feel free to reach out.