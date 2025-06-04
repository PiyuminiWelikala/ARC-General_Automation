from functions import *

class method:

    # Define folder paths
    input_terpenes_folder = 'Data/Terpenes/Input' 
    output_terpenes_folder = 'Data/Terpenes/Database'

    logger.info('<<<<<Starting data transformation for Terpenes>>>>>')

    process_terpens_files(input_terpenes_folder, output_terpenes_folder)

    logger.info('<<<<<Finished data transformation for Terpenes>>>>>')