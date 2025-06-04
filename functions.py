import pandas as pd
import re
import os
import logging
import sys
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)-8s :: %(message)s',
                    filename='log.txt', filemode='a')
logging.getLogger('numexpr').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ==== ETL2 ==== #

def load_name_mapping(dictionary_file_path):
    """
    Load the dictionary Excel file and create a mapping of alternative names to standardized names.
    Arguments:
        dictionary_file_path (str): File path to the dictionary Excel file.
    Returns:
        dict: A dictionary where keys are alternative names and values are standardized names.
    Raises:
        FileNotFoundError: If dictionary file is missing
        Exception: If dictionary loading fails
    """
    try:
        df_dict = pd.read_excel(dictionary_file_path, header=None)
        name_mapping = {}
        for _, row in df_dict.iterrows():
            standardized_name = row[0]
            variations = row[1:].dropna().tolist()
            for variation in variations:
                name_mapping[variation] = standardized_name
        logger.info("Successfully loaded dictionary from %s", dictionary_file_path)
        return name_mapping
    except FileNotFoundError:
        logger.error("E200: Dictionary file for Alpha-Beta-Acid not found: %s", dictionary_file_path)
        sys.exit("Error: Dictionary file for Alpha-Beta-Acid not found.")
    except Exception as e:
        logger.error("E201: Error loading dictionary: %s", str(e))
        sys.exit("Error: Failed to load dictionary.")

def extract_repetition(value):
    """
    Extract the repetition number from a string if it ends with '_1', '_2', etc.
    Arguments:
        value (str): The input string containing a potential repetition pattern.
    Returns:
        str or None: The extracted repetition number if found, otherwise None.
    """
    match = re.search(r'_(\d+)(?=\s|$)', str(value))
    return match.group(1) if match else None

def extract_dilution_factor(value):
    """
    Extract the dilution factor from a string if it contains a number followed by 'x'.
    Arguments:
        value (str): The input string containing a potential dilution factor.
    Returns:
        str or None: The extracted dilution factor if found, otherwise None.
    """
    match = re.search(r'(\d+)(?=x)', str(value))
    return match.group(1) if match else None

def requires_dilution_factor(value):
    """
    Determine if a given value contains a dilution factor pattern.
    Arguments:
        value (str): The input string to check.
    Returns:
        bool: True if a dilution factor pattern is detected, otherwise False.
    """
    return bool(re.search(r'\d+x', str(value)))

def clean_data(df, name_mapping):
    """
    Clean and process the input DataFrame while retaining relevant columns and headers.
        - Extracts repetition numbers.
        - Extracts and handles dilution factors.
        - Standardizes compound names using the provided name mapping.
        - Removes unnecessary characters (e.g., repetition suffixes, dilution indicators).
        - Drops unnecessary columns based on missing values.
    Arguments:
        df (pd.DataFrame): The input DataFrame containing raw data.
        name_mapping (dict): Dictionary mapping alternative names to standardized names.
    Returns:
        pd.DataFrame: A cleaned DataFrame with processed columns.
    Raises:
        Exception: If the data cleaning process failed
    """
    try:
        df = df.iloc[:, :9]  # Keep columns B:H
        first_column = df.columns[0]
        
        df['Repetition'] = df[first_column].apply(extract_repetition)  # Extract repetition
        df['Dilution Factor'] = df[first_column].apply(extract_dilution_factor)  # Extract dilution factor

        # Check if dilution factor column is needed
        df['Needs Dilution Factor'] = df[first_column].apply(requires_dilution_factor)
        
        df[first_column] = df[first_column].apply(lambda x: re.sub(r'_(\d+)(?=\s|$)', '', str(x)))  # Remove suffix
        df[first_column] = df[first_column].apply(lambda x: re.sub(r' (\d+)x', '', str(x)))  # Remove dilution part
        df[first_column] = df[first_column].apply(lambda x: name_mapping.get(str(x), str(x)))  # Standardize names
        
        # Rename the first column to "Compound Name"
        df = df.rename(columns={first_column: "Compound Name"})
        
        if df['Needs Dilution Factor'].any():  # If any row requires dilution factor
            df.insert(1, 'Repetition', df.pop('Repetition'))
            df.insert(2, 'Dilution Factor', df.pop('Dilution Factor'))
        else:
            df.insert(1, 'Repetition', df.pop('Repetition'))
            df = df.drop(columns=['Dilution Factor'])
        
        df = df.drop(columns=['Needs Dilution Factor'])
        
        # Drop rows with more than 3 missing values
        df = df.dropna(thresh=len(df.columns) - 3, axis=0)

        # Check for non-numeric values in all columns except 'Compound Name', 'Collection Date', and 'Extraction Date'
        columns_to_check = [col for col in df.columns if col not in ['Compound Name', 'Collection Date', 'Extraction Date']]
        non_numeric = df[columns_to_check].apply(lambda col: col.map(lambda x: pd.to_numeric(x, errors='coerce'))).isna()

        if non_numeric.any().any():
            logger.error("E205: Non-numeric values found in numeric-only columns.")
            raise Exception("Non-numeric data found in numeric-only columns.")

        logger.info("Successfully cleaned data.")
        return df
    except Exception as e:
        logger.error("E202: Error cleaning Alpha-Beta-Acid data: %s", str(e))
        raise

def process_excel_files(individual_batches_path, modified_batches_path, dictionary_file_path):
    """
    Process all Excel files in the specified folder.
        - Loads the name mapping from the dictionary file.
        - Iterates through each Excel file in the folder.
        - Reads and cleans data from each sheet.
        - Saves the cleaned data to a new output Excel file.
    Arguments:
        individual_batches_path (str): Path to the folder containing input Excel files.
        modified_batches_path (str): Path to the folder where the processed output will be saved.
        dictionary_file_path (str): Path to the dictionary file
    Raises:
        Exception: If the processing failed
    """
    name_mapping = load_name_mapping(dictionary_file_path)
    files_processed = 0

    try:
        if not os.path.exists(modified_batches_path):
            os.makedirs(modified_batches_path)

        files = [f for f in os.listdir(individual_batches_path) if f.endswith('.xlsx')]
        if not files:
            logger.error("E203: No Excel files found in %s", individual_batches_path)
            sys.exit("Error: No Excel files found.")

        for file_name in files:
            file_path = os.path.join(individual_batches_path, file_name)
            xls = pd.ExcelFile(file_path)

            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name, header=0)  # Keep headers
                df_cleaned = clean_data(df, name_mapping)
                output_file = os.path.join(modified_batches_path, f"Modified_{os.path.splitext(file_name)[0]}.xlsx")
                df_cleaned.to_excel(output_file, index=False)

                logger.info("Processed and saved: %s", output_file)
                print(f"Processed file saved as: {output_file}")
                files_processed += 1

        logger.info("Processing completed. Total files processed: %d", files_processed)
    except Exception as e:
        logger.error("E203: Error processing files: %s", str(e))
        sys.exit("Error: Processing failed.")

def process_alpha_beta_acid_files(modified_batches_path, output_alpha_beta_acid):
    """
    Process Excel files containing Alpha and Beta Acid data.
        - Reads input files from the specified folder.
        - Extracts relevant chemical components and concentrations.
        - Saves the processed data in an output folder.
    Arguments:
        modified_batches_path (str): Path to the folder containing input Excel files.
        output_alpha_beta_acid (str): Path to the folder where processed files will be saved.
    """
    try:
        # Check if input folder exists
        if not os.path.exists(modified_batches_path):
            logger.error("E204: Input folder does not exist: %s", modified_batches_path)
            sys.exit(f"Input folder does not exist: {modified_batches_path}")  # Exit on error

        # Get list of all Excel files in the input folder
        files = [f for f in os.listdir(modified_batches_path) if f.endswith(('.xlsx', '.xls'))]
        
        if not files:
            logger.warning("No Excel files found in the input folder: %s", modified_batches_path)
            print("Warning: No Excel files found.")
            sys.exit("No Excel files found in the input folder. Exiting the system.")  # Exit if no files

        # Ensure output folder exists
        os.makedirs(output_alpha_beta_acid, exist_ok=True)
        
        # Process each file
        for file in files:
            file_path = os.path.join(modified_batches_path, file)
            
            try:
                # Read the Excel file
                df = pd.read_excel(file_path)
                
                # Check if required columns exist
                required_columns = ['Compound Name', 'Repetition', 'α-acid Cohumulone', 'α-acid n-+Adhumulone', 'β-acid Colupulone', 
                                    'β-acid n-+Adlupulone']

                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logger.warning("Skipping file %s due to missing columns: %s", file, missing_columns)
                    continue
                
                # Check if 'Dilution Factor' column exists
                dilution_factor_column_exists = 'Dilution Factor' in df.columns
                
                # Create an empty list to store output rows
                output_rows = []
                
                # Iterate through each row of the input dataframe
                for _, row in df.iterrows():
                    sample_name = row['Compound Name']
                    repetition = row['Repetition']
                    start_timestamp = row['Collection Date']
                    end_timestamp = row['Extraction Date']
                    
                    # Get the chemical components and concentrations
                    components = ['α-acid Cohumulone', 'α-acid n-+Adhumulone', 'β-acid Colupulone', 'β-acid n-+Adlupulone']
                    for component in components:
                        # Prepare each output row
                        output_row = {
                            'Sample Name': sample_name,
                            'Chemical Component': component,
                            'Repetition': repetition,
                            'Dilution Factor': row['Dilution Factor'] if dilution_factor_column_exists else '',
                            'Concentration': row.get(component, None),
                            'Collection Date': start_timestamp,
                            'Extraction Date': end_timestamp
                        }
                        output_rows.append(output_row)
                
                # Convert the output rows into a DataFrame
                output_df = pd.DataFrame(output_rows)
                
                # Define output file path
                output_file_path = os.path.join(output_alpha_beta_acid, f"Output_{file}")
                
                # Write the processed data to a new Excel file
                output_df.to_excel(output_file_path, index=False)
                logger.info("Processed and saved: %s", output_file_path)
                print(f"Processed file saved as: {output_file_path}")
                
            except Exception as e:
                logger.error("E203: Error processing file %s: %s", file, str(e))
                print(f"Error processing {file}: {str(e)}")
                sys.exit(f"Error processing {file}. Exiting the system.")  # Exit on error
    
    except Exception as e:
        logger.critical("Fatal error during processing: %s", str(e))
        print(f"Fatal error: {str(e)}")
        sys.exit(f"Fatal error: {str(e)}. Exiting the system.")  # Exit on error

def process_canabis_files(input_cannabis_folder, output_cannabis_folder):
    """
    Processes Excel files in the input folder by extracting relevant data and saving it to the output folder.
    Each file is read, cleaned, and saved in a new format with relevant details.
    Arguments:
        input_cannabis_folder (str): Path to the folder containing input Excel files.
        output_cannabis_folder (str): Path to the folder where the processed output will be saved.
    """
    try:
        # Check if the input folder exists
        if not os.path.exists(input_cannabis_folder):
            logger.error("E204: Input folder does not exist: %s", input_cannabis_folder)
            sys.exit(f"Input folder not found: {input_cannabis_folder}")  # Exit on error
        
        # Check if the output folder exists, create it if it doesn't
        if not os.path.exists(output_cannabis_folder):
            os.makedirs(output_cannabis_folder)
            logger.info("Output folder created: %s", output_cannabis_folder)

        # Loop through all files in the input folder
        for file_name in os.listdir(input_cannabis_folder):
            if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
                file_path = os.path.join(input_cannabis_folder, file_name)
                try:
                    # Load the Excel file and select the sheet named "Mean (%)"
                    df = pd.read_excel(file_path, sheet_name="Mean (%)")
                    
                    # Get sample names from the first row (excluding the first cell)
                    sample_names = df.columns[1:].tolist()  # Exclude the first column (Chemical Component)
                    
                    # Initialize an empty list to store the output data
                    output_data = []

                    start_date_row = df[df.iloc[:, 0] == "Collection Date"]
                    end_date_row = df[df.iloc[:, 0] == "Extraction Date"]
                    
                    start_dates = start_date_row.iloc[:, 1:].values.flatten() if not start_date_row.empty else [None] * len(sample_names)
                    end_dates = end_date_row.iloc[:, 1:].values.flatten() if not end_date_row.empty else [None] * len(sample_names)
                    
                    # Loop through each row in the dataframe
                    for _, row in df.iterrows():
                        # The first column is the Chemical Component
                        chemical_component = row.iloc[0]  # First column in the row is the component
                        if chemical_component in ["Collection Date", "Extraction Date"]:
                            continue
                        
                        # Loop through each sample (excluding the first column which is the component)
                        for i, sample_name in enumerate(sample_names):
                            concentration = row[sample_name]
                            
                            # Clean the sample name by removing "Mean" and stripping extra spaces
                            cleaned_sample_name = sample_name.replace("Mean", "").strip()
                            
                            # Skip rows where the concentration is 'LOD' or 'LOQ'
                            if pd.notna(concentration) and concentration not in ['LOD', 'LOQ']:
                                # Add the row to output data
                                output_data.append({
                                    "Sample Name": cleaned_sample_name,
                                    "Chemical Component": chemical_component,
                                    "Concentration": concentration,
                                    "Collection Date": start_dates[i],
                                    "Extraction Date": end_dates[i]
                                })
                        
                    # Create the output DataFrame
                    output_df = pd.DataFrame(output_data)

                    # Check for non-numeric values in 'Concentration'
                    columns_to_check = ['Concentration']
                    non_numeric = output_df[columns_to_check].apply(lambda col: col.map(lambda x: pd.to_numeric(x, errors='coerce'))).isna()
                    
                    if non_numeric.any().any():
                        logger.error("E205: Non-numeric values found in numeric-only columns.")
                        raise Exception("Non-numeric data found in numeric-only columns.")

                    # Save the output DataFrame to a new Excel file
                    output_file_name = f"Output_{file_name}"
                    output_file_path = os.path.join(output_cannabis_folder, output_file_name)
                    output_df.to_excel(output_file_path, index=False)
                    
                    logger.info("Processed file saved as %s", output_file_name)
                    print(f"Processed file saved as {output_file_name}")
                    
                except Exception as e:
                    logger.error("E203: Error processing file %s: %s", file_name, str(e))
                    print(f"Error processing file {file_name}. Check log for details.")
                    sys.exit(f"Error processing file {file_name}. Exiting the system.")  # Exit on error
    
    except Exception as e:
        logger.error("E203: Error in processing files: %s", str(e))
        print("An error occurred. Check the log for details.")
        sys.exit("An error occurred. Exiting the system.")  # Exit on error

def process_terpens_files(input_terpenes_folder, output_terpenes_folder):
    """
    Processes Excel files in the input folder by extracting relevant data and saving it to the output folder.
    Each file is read, cleaned, and saved in a new format with relevant details.
    Arguments:
        input_terpenes_folder (str): Path to the folder containing input Excel files.
        output_terpenes_folder (str): Path to the folder where the processed output will be saved.
    """
    try:
        # Check if the input folder exists
        if not os.path.exists(input_terpenes_folder):
            logger.error("E204: Input folder does not exist: %s", input_terpenes_folder)
            sys.exit(f"Input folder not found: {input_terpenes_folder}")  # Exit on error
        
        # Check if the output folder exists, create it if it doesn't
        if not os.path.exists(output_terpenes_folder):
            os.makedirs(output_terpenes_folder)
            logger.info("Output folder created: %s", output_terpenes_folder)

        # Loop through all files in the input folder
        for file_name in os.listdir(input_terpenes_folder):
            if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
                file_path = os.path.join(input_terpenes_folder, file_name)
                try:
                    # Load the Excel file and select the sheet named "Mean (%)"
                    df = pd.read_excel(file_path, sheet_name="Mean (ml 100g^-1)")
                    
                    # Get sample names from the first row (excluding the first cell)
                    sample_names = df.columns[1:].tolist()  # Exclude the first column (Chemical Component)
                    
                    # Initialize an empty list to store the output data
                    output_data = []

                    # Extract start and end date rows
                    start_date_row = df[df.iloc[:, 0] == "Collection Date"]
                    end_date_row = df[df.iloc[:, 0] == "Extraction Date"]
                    
                    # Get start and end dates for each sample
                    start_dates = start_date_row.iloc[:, 1:].values.flatten() if not start_date_row.empty else [None] * len(sample_names)
                    end_dates = end_date_row.iloc[:, 1:].values.flatten() if not end_date_row.empty else [None] * len(sample_names)
                    
                    # Loop through each row in the dataframe
                    for _, row in df.iterrows():
                        # The first column is the Chemical Component
                        chemical_component = row.iloc[0]  # First column in the row is the component
                        if chemical_component in ["Collection Date", "Extraction Date"]:
                            continue
                        
                        # Loop through each sample (excluding the first column which is the component)
                        for i, sample_name in enumerate(sample_names):
                            concentration = row[sample_name]
                            
                            # Clean the sample name by removing "Mean" and stripping extra spaces
                            cleaned_sample_name = sample_name.replace("Mean", "").strip()
                            
                            # Skip rows where the concentration is 'LOD' or 'LOQ' or 'HLOQ'
                            if pd.notna(concentration) and concentration not in ['LOD', 'LOQ', 'HLOQ']:
                                # Add the row to output data
                                output_data.append({
                                    "Sample Name": cleaned_sample_name,
                                    "Chemical Component": chemical_component,
                                    "Concentration": concentration,
                                    "Collection Date": start_dates[i],
                                    "Extraction Date": end_dates[i]
                                })
                    
                    # Create the output DataFrame
                    output_df = pd.DataFrame(output_data)

                    # Check for non-numeric values in 'Concentration'
                    columns_to_check = ['Concentration']
                    non_numeric = output_df[columns_to_check].apply(lambda col: col.map(lambda x: pd.to_numeric(x, errors='coerce'))).isna()
                    
                    if non_numeric.any().any():
                        logger.error("E205: Non-numeric values found in numeric-only columns.")
                        raise Exception("Non-numeric data found in numeric-only columns.")
                    
                    # Save the output DataFrame to a new Excel file
                    output_file_name = f"Output_{file_name}"
                    output_file_path = os.path.join(output_terpenes_folder, output_file_name)
                    output_df.to_excel(output_file_path, index=False)
                    
                    logger.info("Processed file saved as %s", output_file_name)
                    print(f"Processed file saved as {output_file_name}")
                    
                except Exception as e:
                    logger.error("E203: Error processing file %s: %s", file_name, str(e))
                    print(f"Error processing file {file_name}. Check log for details.")
                    sys.exit(f"Error processing file {file_name}. Exiting the system.")  # Exit on error
    
    except Exception as e:
        logger.error("E203: Error in processing files: %s", str(e))
        print("An error occurred. Check the log for details.")
        sys.exit("An error occurred. Exiting the system.")  # Exit on error

# ==== ETL3 ==== #

# Database connection configuration
DB_CONFIG = {
    "dbname": "ARC_ETL",
    "user": "postgres",
    "password": "1234",
    "host": "localhost",
    "port": "5433"
}

def clean_column_name(name):
    """
    Clean a column name by stripping whitespace and replacing spaces with underscores.
    Args:
        name (str): Original column name.
    Returns:
        str: Cleaned column name.
    """
    return name.strip().replace(" ", "_")

def exit_on_failure(msg):
    """
    Log an error message and terminate the script.
    Args:
        msg (str): Error message to log.
    Raises:
        SystemExit: Exits the script with error status.
    """
    logger.error(msg)
    sys.exit(1)

def get_unique_values(paths):
    """
    Extract unique sample names and chemical components from Excel files in specified directories.
    Args:
        paths (list[str]): List of directory paths containing Excel files.
    Returns:
        tuple[set[str], set[str]]: Unique sample names and chemical components.
    Raises:
        Exception: If extraction fails for any file.
    """
    logger.info("Extracting sample names and chemical components")
    sample_names = set()
    chemical_components = set()
    
    try:
        for path in paths:
            if not os.path.exists(path):
                logger.warning(f"Directory does not exist: {path}")
                continue

            # Iterate through Excel files in the path
            for file in os.listdir(path):
                if file.endswith((".xlsx", ".xls")):
                    file_path = os.path.join(path, file)
                    logger.info(f"Reading file: {file_path}")
                    df = pd.read_excel(file_path, dtype=str)

                    # Collect unique values from the relevant columns
                    if "Sample Name" in df.columns:
                        sample_names.update(df["Sample Name"].dropna().unique())
                    if "Chemical Component" in df.columns:
                        chemical_components.update(df["Chemical Component"].dropna().unique())
    except Exception as e:
        exit_on_failure(f"E300: Failed to extract unique sample names/chemical compounds : {e}")
    
    logger.info(f"Found {len(sample_names)} sample names and {len(chemical_components)} chemical components.")
    return sample_names, chemical_components

def get_instruments_and_procedures(instrument_path, procedure_path):
    """
    Extract unique instrument and procedure names from given Excel files.
    Args:
        instrument_path (str): File path to instrument data Excel file.
        procedure_path (str): File path to procedure data Excel file.
    Returns:
        tuple[set[str], set[str]]: Unique instrument and procedure names.
    Raises:
        Exception: If extraction fails due to file read errors.
    """
    logger.info("Extracting instrument and procedure names")
    instrument_names = set()
    procedure_names = set()

    try:
        if os.path.exists(instrument_path):
            df = pd.read_excel(instrument_path, dtype=str)
            df.columns = [clean_column_name(col) for col in df.columns]
            instrument_names.update(df["dimension_instrument_Name"].dropna().unique())

        if os.path.exists(procedure_path):
            df = pd.read_excel(procedure_path, dtype=str)
            df.columns = [clean_column_name(col) for col in df.columns]
            procedure_names.update(df["dim_procedure_Name"].dropna().unique())
    except Exception as e:
        exit_on_failure(f"E301: Failed to extract instruments/procedures: {e}")

    logger.info(f"Found {len(instrument_names)} instruments and {len(procedure_names)} procedures.")
    return instrument_names, procedure_names

def insert_missing_values(sample_names, chemical_components, instrument_names, procedure_names):
    """
    Insert any missing values into the dimension tables (sample, chemical, instrument, procedure).
    Args:
        sample_names (set[str]): Set of unique sample names.
        chemical_components (set[str]): Set of unique chemical components.
        instrument_names (set[str]): Set of unique instrument names.
        procedure_names (set[str]): Set of unique procedure names.
    Returns:
        None
    Raises:
        Exception: If insertion fails or database operation fails.
    """
    logger.info("Inserting missing values into dimension tables.")

    conn = None
    cursor = None

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Helper function to fetch existing values from DB
        def fetch_existing(query):
            cursor.execute(query)
            return {row[0] for row in cursor.fetchall()}
        
        # Get current values to avoid duplicates
        existing_samples = fetch_existing("SELECT dim_sample_name FROM dimension.sample;")
        existing_chemicals = fetch_existing("SELECT dim_chemicalcompound_name FROM dimension.chemicalcompound;")
        existing_instruments = fetch_existing("SELECT dimension_instrument_name FROM dimension.instrument;")
        existing_procedures = fetch_existing("SELECT dim_procedure_name FROM dimension.procedure;")

        # Identify missing entries
        new_samples = sample_names - existing_samples
        new_chemicals = chemical_components - existing_chemicals
        new_instruments = instrument_names - existing_instruments
        new_procedures = procedure_names - existing_procedures

        now = datetime.now()

        # Helper function to perform bulk inserts with logging
        def insert_records(query, values):
            if values:
                execute_values(cursor, query, values)
                logger.info(f"Inserted {len(values)} new records.")

        insert_records(""" 
            INSERT INTO dimension.sample (dim_sample_name, dim_sample_creationdate)
            VALUES %s;
        """, [(name, now) for name in new_samples])

        insert_records(""" 
            INSERT INTO dimension.chemicalcompound (dim_chemicalcompound_name, dim_chemicalcompount_creationdate)
            VALUES %s;
        """, [(name, now) for name in new_chemicals])

        insert_records(""" 
            INSERT INTO dimension.instrument (dimension_instrument_name, dimension_instrument_creationdate)
            VALUES %s;
        """, [(name, now) for name in new_instruments])

        insert_records(""" 
            INSERT INTO dimension.procedure (dim_procedure_name, dim_procedure_creationdate)
            VALUES %s;
        """, [(name, now) for name in new_procedures])

        conn.commit()
        logger.info("All missing values from sample, chemical compound, instrument, and procedure inserted successfully.")

    except Exception as e:
        logger.exception("Error inserting missing values.")
        if conn:
            conn.rollback()
        exit_on_failure(f"E302: Insertion failed: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def load_excel(file_path):
    """
    Load an Excel file and return a cleaned DataFrame with standardized column names.
    Args:
        file_path (str): Full file path to the Excel file.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If reading Excel fails.
    """
    logger.info(f"Loading Excel file: {file_path}")
    try:
        df = pd.read_excel(file_path, dtype=str)
        df.columns = [clean_column_name(col) for col in df.columns]
        return df
    except Exception as e:
        exit_on_failure(f"E303: Failed to load Excel file: {e}")

def update_batch_records(df_batch):
    """
    Insert new or update existing batch records in the batch dimension table.
    Args:
        df_batch (pd.DataFrame): DataFrame containing batch information.
    Returns:
        None
    Raises:
        Exception: If database update or insert fails.
    """
    logger.info("Updating/inserting batch records...")

    conn = None
    cursor = None

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Fetch all existing batches and their current metadata
        cursor.execute("SELECT dim_batch_name, dim_batch_description, dim_batch_extractiondate, dim_batch_processingdate FROM dimension.batch;")
        existing_batches = {row[0]: row[1:] for row in cursor.fetchall()}

        df_batch = df_batch.fillna("")
        new_batches, updates = [], []
        now = datetime.now()

        # Compare with existing to detect new or changed records
        for _, row in df_batch.iterrows():
            batch_name = row["dim_batch_name"].strip()
            batch_desc = row.get("dim_batch_description", "").strip()
            extraction_date = pd.to_datetime(row.get("dim_batch_extractionDate", None), errors='coerce')
            processing_date = pd.to_datetime(row.get("dim_batch_processingDate", None), errors='coerce')

            extraction_date = extraction_date if pd.notna(extraction_date) else None
            processing_date = processing_date if pd.notna(processing_date) else None

            if batch_name in existing_batches:
                if (batch_desc, extraction_date, processing_date) != existing_batches[batch_name]:
                    updates.append((batch_desc, extraction_date, processing_date, now, batch_name))
            else:
                new_batches.append((batch_name, batch_desc, extraction_date, processing_date, now))

        # Perform insertions and updates
        if new_batches:
            execute_values(cursor, """
                INSERT INTO dimension.batch (dim_batch_name, dim_batch_description, dim_batch_extractiondate, dim_batch_processingdate, dim_batch_creationdate)
                VALUES %s;
            """, new_batches)
            logger.info(f"Inserted {len(new_batches)} new batches.")

        for update in updates:
            cursor.execute("""
                UPDATE dimension.batch
                SET dim_batch_description = %s, dim_batch_extractiondate = %s, dim_batch_processingdate = %s, dim_batch_modifieddate = %s
                WHERE dim_batch_name = %s;
            """, update)
        logger.info(f"Updated {len(updates)} existing batches.")

        conn.commit()
    except Exception as e:
        logger.exception("Error updating batch records.")
        if conn:
            conn.rollback()
        exit_on_failure(f"E304: Batch update failed: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def update_sop_records(df_sop):
    """
    Insert new SOP records with auto-incremented version numbers if not already existing.
    Args:
        df_sop (pd.DataFrame): DataFrame containing SOP data.
    Returns:
        None
    Raises:
        Exception: If SOP insert or version handling fails.
    """
    logger.info("Inserting SOP records.")

    conn = None
    cursor = None

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Get existing SOPs and latest version info
        cursor.execute("SELECT dim_procedure_id, dim_sop_description FROM dimension.sop;")
        existing_sop_keys = {
            (str(row[0]).strip(), str(row[1]).strip())
            for row in cursor.fetchall()
        }

        cursor.execute("SELECT dim_procedure_id, MAX(dim_sop_version) FROM dimension.sop GROUP BY dim_procedure_id;")
        db_max_versions = {row[0]: row[1] for row in cursor.fetchall()}

        new_sops = []
        local_version_tracker = db_max_versions.copy()
        now = datetime.now()

        # Loop through input SOPs and add if not already present
        for _, row in df_sop.iterrows():
            procedure_id_str = str(row["dim_procedure_id"]).strip()
            sop_description = str(row["dim_sop_description"]).strip()
            key = (procedure_id_str, sop_description)

            procedure_id = int(procedure_id_str)  # Use int for version tracking

            if key not in existing_sop_keys:
                new_version = local_version_tracker.get(procedure_id, 0) + 1
                local_version_tracker[procedure_id] = new_version
                new_sops.append((procedure_id, new_version, sop_description, now))

        # Insert new SOP entries
        if new_sops:
            execute_values(cursor, """
                INSERT INTO dimension.sop (dim_procedure_id, dim_sop_version, dim_sop_description, dim_sop_creationdate)
                VALUES %s;
            """, new_sops)
            logger.info(f"Inserted {len(new_sops)} new SOP records.")

        conn.commit()
    except Exception as e:
        logger.exception("Error updating SOP records.")
        if conn:
            conn.rollback()
        exit_on_failure(f"E305: SOP update failed: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
