from functions import *

class method:

    # Define the paths for Alpha_Beta_Acid, Cannabis and Terpenes
    paths = [
        "Data/Alpha_Beta_Acid/Database",
        "Data/Cannabis/Database",
        "Data/Terpenes/Database"
    ]
    # Define the paths for Instrument, Procedure, Batch and SOP
    instrument_path = "Data/instrument.xlsx"
    procedure_path = "Data/procedure.xlsx"
    batch_path = "Data/batch.xlsx"
    sop_path = "Data/sop.xlsx"

    logger.info('<<<<<Starting ETL3>>>>>')

    logger.info('<<<<<Get Unique values for Sample, Chemical Component, Instrument and Procedure>>>>>')

    sample_names, chemical_components = get_unique_values(paths)
    instrument_names, procedure_names = get_instruments_and_procedures(instrument_path, procedure_path)

    logger.info('<<<<<Starting Insert unique values from Sample, Chemical Component, Instrument and Procedure to the Database>>>>>')

    insert_missing_values(sample_names, chemical_components, instrument_names, procedure_names)

    logger.info('<<<<<Done Inserting unique values from Sample, Chemical Component, Instrument and Procedure to the Database>>>>>')

    logger.info('<<<<<Starting Insert or Update unique values from Batch to the Database>>>>>')

    df_batch = load_excel(batch_path)
    update_batch_records(df_batch)

    logger.info('<<<<<Done Insert or Update unique values from Batch to the Database>>>>>')

    logger.info('<<<<<Starting Insert of unique values from SOP to the Database>>>>>')

    df_sop = load_excel(sop_path)
    update_sop_records(df_sop)

    logger.info('<<<<<Done Insert of unique values from SOP to the Database>>>>>')

    logger.info('<<<<<Finishing ETL3>>>>>')