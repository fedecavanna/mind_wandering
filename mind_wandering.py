""" 
Rasgo:
    BFI: E A C N O
    STAI-R: Trait anxiety
    TAS: Absorption
    SRIS: Self-Reflection, Need for S-R, Insight
    OSIVQ: Object, Spatial, Verbal (types)
    VVIQ: Vividness imagery
    
    MWQ: Wandering

Estado:
    STAI-E: State anxiety 
"""

def prepro_scales(df):
    # algunas columnas vienen con espacios, las acomodo
    df.columns = df.columns.str.strip()
    
    return df

def score_scales(df): 
    import score_lib as score_lib
    import pandas as pd
    
    df_scored = pd.DataFrame()
    
    # ID, Condición y Dosis
    df_scored['ID'] = df.loc[:,'ID']
    
    # Primero obtengo los rangos de columnas para cada escala
    # Nota: range() toma hasta n-1, por lo cual tengo que extenderme un índice más.
    bfi_cols = list(range(2,45+1)) 
    staie_cols = list(range(46,65+1))
    stair_cols = list(range(66,85+1))
    tas_cols = list(range(86,119+1))
    mwq_cols = list(range(120,124+1))
    osivq_cols = list(range(125,169+1))    
    vviq_cols =  list(range(170,184+1))
    sris_cols = list(range(185,204+1))

    # Reemplazo los valores de tipo texto por valores numéricos
    # BFI viene numerada
    df.iloc[:,staie_cols] = score_lib.replace_stai_e(df.iloc[:,staie_cols]) # STAI-E
    df.iloc[:,stair_cols] = score_lib.replace_stai_r(df.iloc[:,stair_cols]) # STAI-R
    df.iloc[:,tas_cols] = score_lib.replace_tas(df.iloc[:,tas_cols]) # TAS
    df.iloc[:,mwq_cols] = score_lib.replace_mwq(df.iloc[:,mwq_cols]) # MWQ
    df.iloc[:,sris_cols] = score_lib.replace_sris(df.iloc[:,sris_cols]) # SRIS
    df.iloc[:,osivq_cols] = score_lib.replace_osivq(df.iloc[:,osivq_cols]) # OSIVQ
    df.iloc[:,vviq_cols] = score_lib.replace_vviq(df.iloc[:,vviq_cols]) # VVIQ
    
    # Scoreo las escalas
    df_scored = df_scored.join(score_lib.score_bfi(df.iloc[:,bfi_cols])) # BFI
    df_scored = df_scored.join(score_lib.score_stai_e(df.iloc[:,staie_cols])) # STAI-E
    df_scored = df_scored.join(score_lib.score_stai_r(df.iloc[:,stair_cols])) # STAI-R
    df_scored = df_scored.join(score_lib.score_tas(df.iloc[:,tas_cols])) # TAS
    df_scored = df_scored.join(score_lib.score_mwq(df.iloc[:,mwq_cols])) # MWQ
    df_scored = df_scored.join(score_lib.score_sris(df.iloc[:,sris_cols])) # SRIS
    df_scored = df_scored.join(score_lib.score_osivq(df.iloc[:,osivq_cols])) # OSIVQ
    df_scored = df_scored.join(score_lib.score_vviq(df.iloc[:,vviq_cols])) # VVIQ
    return df_scored

def proc_correlations(df, savedir=''):        
    import main_lib as mlib
    # MATRIZ DE CORRELACIÓN
    df_subset = df.loc[:,['BFI-E','BFI-A','BFI-C','BFI-N','BFI-O','STAI-R',
                        'TAS','SRIS-SelfReflection','SRIS-NeedSR','SRIS-Insight','OSIVQ-Object','OSIVQ-Spacial','OSIVQ-Verbal','VVIQ','MWS']]
    cols_labels = ['BFI-E','BFI-A','BFI-C','BFI-N','BFI-O','STAI-R',
                    'TAS','SRIS-SR','SRIS-N','SRIS-I','OSIVQ-O','OSIVQ-S','OSIVQ-V','VVIQ','MWS']
    rows_labels = cols_labels    
    method = 'pearson' #'spearman'
    mlib.plot_corr_matrix(df_subset, "", cols_labels, rows_labels, p_value=0.05, r_threshold=0.3, mult_comparison=True, method=method, sq_size=12, save_dir=savedir, file_name="corr_0")              
    return

# -------------------------------------------------------------------------------------------------------------------------------------------     
# MAIN
def main():
    import sys
    import pandas as pd
    
    # agrego el path para main_lib y score_lib
    sys.path.append('../_common')
        
    filepath = r'C:\Users\Fede\Desktop\Drive\Labo\Mind wandering'
    raw_df = pd.read_csv(r'{}\escalas.csv'.format(filepath))

    try:
        preprocess_data = False
        if preprocess_data:
            # cargo las db y preproceso los datos
            main_df = prepro_scales(raw_df)
                        
            # puntúo las escalas
            main_df = score_scales(main_df)

            # guardo escalas scoreadas
            main_df.to_csv(r'{}\escalas_SCORED.csv'.format(filepath))
            
        # Cargo todos los datos utilizables
        main_df = pd.read_csv(r'{}\escalas_SCORED.csv'.format(filepath))

        # proc_correlations(main_df)

        print('Run OK')
        sys.exit(0)    
    except Exception as e:
        print(e)
        sys.exit(1)

if __name__== "__main__":
  main()