from functions import *

class method:

    # Define folder paths and dictionary file
    individual_batches_path = "Data/Alpha_Beta_Acid/Individual Batches"
    modified_batches_path = "Data/Alpha_Beta_Acid/Output"
    dictionary_file_path = "Data/Alpha_Beta_Acid/Dictionary.xlsx"
    output_alpha_beta_acid = 'Data/Alpha_Beta_Acid/Database'

    logger.info('<<<<<Starting data modification for Alpha-Beta Acid>>>>>')

    process_excel_files(individual_batches_path, modified_batches_path, dictionary_file_path)

    logger.info('<<<<<Finished data modification for Alpha-Beta Acid>>>>>')

    logger.info('<<<<<Starting data transformation for Alpha-Beta Acid>>>>>')

    process_alpha_beta_acid_files(modified_batches_path, output_alpha_beta_acid)

    logger.info('<<<<<Finished data transformation for Alpha-Beta Acid>>>>>')