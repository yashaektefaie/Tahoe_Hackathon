from agent.utils import *
from agent.prompt import *
import anndata
import gradio as gr
from gradio import ChatMessage
import re
import pandas as pd
import pathlib
import numpy as np

class SigSpace(Basic_Agent):
    def __init__(self, config_path:str):
        super().__init__(config_path)
        self.conversation = []
        self.system_prompt = Agent_Prompt
        self.conversation = []
        self.conversation.append({"role": "system", "content": self.system_prompt})
        
        # initialize data for jump
        jump_path = pathlib.Path("/home/ubuntu/giovanni/data")
        self.jump_tahoe_drug_metadata = pd.read_csv(jump_path/"drug_metadata_inchikey.csv")
        self.jump_similarity_score = pd.read_csv(jump_path/"compound_genetic_perturbation_cosine_similarity_inchikey.csv")
        
        # Load PRISM IC50 matrix
        prism_data_path = pathlib.Path("/home/ubuntu/sid/Hackathon_Tahoe/data")
        self.ic50 = pd.read_csv(prism_data_path / "Tahoe_PRISM_cell_by_drug_ic50_matrix_named.csv", index_col=0)
        self.ic50.columns = self.ic50.columns.str.lower()

        nci60_path = pathlib.Path("/home/ubuntu/ishita/tahoe/")
        self.lc50 = pd.read_csv(nci60_path / "filtered_results.csv")
        # Filter out rows where CELL is nan
        self.lc50 = self.lc50[self.lc50['CELL'].notna()]

        # Load full Tahoe metadata
        tahoe_path = pathlib.Path("/home/ubuntu/rohit/data")
        self.tahoe_cell_meta = pd.read_csv(tahoe_path / "cell_line_metadata.csv")
        self.tahoe_drug_meta = pd.read_csv(tahoe_path / "drug_metadata.csv")
        self.tahoe_vision_scores = anndata.read_h5ad(tahoe_path / "tahoe_vision_scores.h5ad")
        
        # Load PRISM subset of Tahoe metadata
        self.prism_tahoe_cell_meta = pd.read_csv(prism_data_path / "Tahoe_PRISM_matched_cell_metadata_final.csv")
        self.prism_tahoe_drug_meta = pd.read_csv(prism_data_path / "Tahoe_PRISM_matched_drug_metadata_final.csv")

        # Build cell line common name to depmap_id map (strip whitespace and case)
        self.cell_name_to_depmap = {
            row["cell_name"].strip(): row["Cell_ID_DepMap"]
            for _, row in self.prism_tahoe_cell_meta.iterrows()
        }

        self.cell_name_to_depmap_lc50 = {
            row["clean"].strip(): row["cell_line_name"]
            for _, row in self.lc50.iterrows()
        }

    
    def initialize_conversation(self, message, conversation=None, history=None):
        if conversation is None:
            conversation = []

        conversation.append({"role": "system", "content" : Agent_Prompt})
        
        if history is not None:
            if len(history) == 0:
                conversation = []
                print("clear conversation successfully")
            else:
                for i in range(len(history)):
                    if history[i]['role'] == 'user':
                        if i-1 >= 0 and history[i-1]['role'] == 'assistant':
                            conversation.append(
                                {"role": "assistant", "content": history[i-1]['content']})
                        conversation.append(
                            {"role": "user", "content": history[i]['content']})
                    if i == len(history)-1 and history[i]['role'] == 'assistant':
                        conversation.append(
                            {"role": "assistant", "content": history[i]['content']})

        conversation.append({"role": "user", "content": message})

        return conversation

    def get_similar_disease(self, disease_name, k_value):
        if disease_name != "Alzheimer's":
            return "FAIL"
        return 'Parkinsons Disease'

    def get_validated_target_jump(self, drug_name):
        print(drug_name)
        try:
            inchikey = self.jump_tahoe_drug_metadata[self.jump_tahoe_drug_metadata.drug.isin([drug_name])]["InChIKey"].values[0]
            similarity_scores = self.jump_similarity_score[self.jump_similarity_score.InChIKey.isin([inchikey])]

            # Count ORF entries with cosine_similarity > 0.2 and < -0.2
            orf_positive = similarity_scores[(similarity_scores.Genetic_Perturbation == 'ORF') & (similarity_scores.cosine_sim > 0.2)].shape[0]
            orf_negative = similarity_scores[(similarity_scores.Genetic_Perturbation == 'ORF') & (similarity_scores.cosine_sim < -0.2)].shape[0]

            # Count CRISPR entries with cosine_similarity > 0.2 and < -0.2
            crispr_positive = similarity_scores[(similarity_scores.Genetic_Perturbation == 'CRISPR') & (similarity_scores.cosine_sim > 0.2)].shape[0]
            crispr_negative = similarity_scores[(similarity_scores.Genetic_Perturbation == 'CRISPR') & (similarity_scores.cosine_sim < -0.2)].shape[0]

            orf_targets = f"ORF: {orf_positive} positive correlations (>0.2), {orf_negative} negative correlations (<-0.2)"
            crispr_targets = f"CRISPR: {crispr_positive} positive correlations (>0.2), {crispr_negative} negative correlations (<-0.2)"

            orf_crispr_targets = orf_targets + " " +crispr_targets

            known_targets_from_jump = self.jump_tahoe_drug_metadata[self.jump_tahoe_drug_metadata.drug.isin([drug_name])]["target_list"].values[0]
            known_targets_output = f"The known targets from the JUMP dataset are: {', '.join(known_targets_from_jump.split('|'))}"
        except Exception as e:
            print(e)
            return "For the drug {drug_name}, we were not able to find the target in the JUMP dataset."
        
        orf_crispr_targets = \
        f"""
        Preturbation description:

        ORF: The ORF perturbation consists of an overexpression of the target gene.
        CRISPR: The CRISPR perturbation consists of a knockout of the target gene.

        Considering the drug "{drug_name}", we expect positive correlations with shared CRISPR targets, 
        and negative correlations with shared ORF targets.
        
        But, the measured correlations are:

        {orf_crispr_targets}

        Furthermore, the JUMP dataset has the following known targets for the drug "{drug_name}":

        {known_targets_output}
        """ 
        return orf_crispr_targets

    def get_ic50_prism(self, drug_name: str, cell_line_name: str):
        drug_name_lower = drug_name.strip().lower()
        cell_line_key = cell_line_name.strip()

        if cell_line_key not in self.cell_name_to_depmap:
            print(f"Cell line name '{cell_line_key}' not found for PRISM data")
            return f"FAIL: Cell line name '{cell_line_key}' not found for PRISM data"

        depmap_id = self.cell_name_to_depmap[cell_line_key]

        if drug_name_lower not in self.ic50.columns:
            print(f"Drug name '{drug_name}' not found in IC50 matrix columns.")
            return f"FAIL: Drug name '{drug_name}' not found in IC50 matrix columns."

        try:
            ic50_val = self.ic50.loc[depmap_id, drug_name_lower]
            if pd.isna(ic50_val):
                print(f"FAIL: IC50 value is missing for '{drug_name}' in cell line '{cell_line_name}' (DepMap ID: {depmap_id}).")
                return f"FAIL: IC50 value is missing for '{drug_name}' in cell line '{cell_line_name}' (DepMap ID: {depmap_id})."

            return (
                f"The IC50 value of {ic50_val:.4f} corresponds to the log10-transformed micromolar concentration "
                f"at which {drug_name} inhibits 50% of viability in the {cell_line_name} cell line "
                f"(DepMap ID: {depmap_id}).\n\n"
                "This value comes from the PRISM Repurposing Secondary Screen, which exposes pooled barcoded cell lines "
                "to drug treatment for 5 days and infers viability from barcode abundance using sequencing.\n\n"
                "The secondary screen includes higher-confidence compound–cell line pairs with improved replicability "
                "compared to the primary screen.\n\n"
                "Lower IC50 values indicate greater sensitivity of the cell line to the drug."
            )
        except KeyError as e:
            print(f"Combination not found: {e}")
            return None


    def clean_cell_line_name(self, name):
        """
        Standardize cell line names for comparison by:
        1. Converting to string (handles any non-string values)
        2. Converting to uppercase
        3. Removing all non-alphanumeric characters
        
        Args:
            name: Cell line name (string or other type)
            
        Returns:
            Cleaned string with only uppercase letters and numbers
        """
        return re.sub(r"[^A-Z0-9]", "", str(name).upper())

    def get_lc50_nci60(self, drug_name: str, cell_line_name: str):
        cell_line_name = cell_line_name.upper()
        cell_line_key = self.clean_cell_line_name(cell_line_name)

        if cell_line_key not in self.cell_name_to_depmap_lc50:
            print(f"Cell line name '{cell_line_key}' not found for NCI60 data")
            return None
        depmap_id = self.cell_name_to_depmap_lc50[cell_line_key]
        print ("Depmap_id", depmap_id)

        # Find the drug in NCI60 dataset
        # Since drugs are in uppercase in the list, convert search term to uppercase
        drug_name_upper = drug_name.strip().upper()

        # Filter rows where the drug name is in the drug column
        # This assumes drugs in each row are comma-separated or in a format that can be searched
        matching_row = self.lc50[self.lc50['drug'].str.contains(drug_name_upper, na=False)]
        print ("Matching row", matching_row)
        if matching_row.empty:
            print(f"Drug name '{drug_name}' not found in NCI60 dataset.")
            return None

        if matching_row.empty:
            raise ValueError(f"Multiple matches found for drug '{drug_name}' in NCI60 dataset.")

        print ("Matching row", matching_row)
        # Get the LC50 value from the matching row
        lc50_val = matching_row.iloc[0]['NLOGLC50']
        lconc_val = matching_row.iloc[0]['LCONC']

        if pd.isna(lc50_val):
            return "LC50 value is missing for '{drug_name}' in cell line '{cell_line_name}' (depmap_id: {depmap_id})."
        
        lc50_output = \
        f"""
        The LC50 value of {lc50_val} represents -log10(LC50), the negative base-10 logarithm of the molar concentration that inhibits 50% of cell growth. 

        Higher LC50 values therefore indicate greater drug potency. 

        The LCONC value of {lconc_val} denotes the maximum log10 molar concentration tested in the dilution series—for example, LCONC = -4 corresponds to 10^-4 M. 

        Both metrics come from the NCI-60 drug screen, which applies a standardized 48-hour exposure assay across all compound–cell-line pairs."
        """
        
        return lc50_output
    
    def load_gene_sets_file(self, file_path):
        """
        Load gene sets from a tab-delimited file where the first column is the gene set name
        and the remaining columns are gene symbols.
        
        Parameters:
        -----------
        file_path : str
            Path to the gene sets file
            
        Returns:
        --------
        dict
            Dictionary mapping gene set names to lists of genes
        """
        gene_sets = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                if parts:
                    set_name = parts[0]
                    genes = [gene for gene in parts[1:] if gene]  # Filter out empty strings
                    gene_sets[set_name] = genes
        return gene_sets

    def get_genes_for_set(self, set_name):
        """
        Get the list of genes for a specific gene set.
        
        Parameters:
        -----------
        set_name : str
            Name of the gene set to query
            
        Returns:
        --------
        list
            List of genes in the gene set, or empty list if set not found
        """
        if not hasattr(self, 'gene_sets'):
            # Load the gene sets file if it hasn't been loaded yet
            self.gene_sets = self.load_gene_sets_file('/home/ubuntu/ishita/msigdb_all_sigs_human_symbols.txt')
        
        return self.gene_sets.get(set_name, [])
            
    def rank_vision_scores(self, drug_name: str, cell_line_name: str, k_value: int):
        self.tahoe_vision_scores.X = (self.tahoe_vision_scores.X - np.mean(self.tahoe_vision_scores.X, axis = 0)) / np.std(self.tahoe_vision_scores.X, axis = 0)

        # subset to the drug / cell line at the highest tested concentration
        filt = (
            (self.tahoe_vision_scores.obs["Cell_Name_Vevo"] == cell_line_name)
            & (self.tahoe_vision_scores.obs["drug"] == drug_name)
        )
        filtered_scores = self.tahoe_vision_scores[filt]
        if filtered_scores.n_obs == 0:
            return "VISION scores not found for this drug–cell-line combination."

        filtered_scores = filtered_scores[
            filtered_scores.obs["concentration"] == filtered_scores.obs["concentration"].max()
        ]

        # pick top-|score| gene sets
        top_idx = np.argsort(-np.abs(filtered_scores.X[0]))[:k_value]
        gene_sets = filtered_scores.var.index[top_idx].tolist()
        scores    = filtered_scores.X[0, top_idx].tolist()

        # build the narrative
        header = (
            "VISION scores are single-cell gene-set enrichment values computed by the "
            "VISION algorithm (DeTomaso & Yosef 2021). Positive scores indicate relative "
            "up-regulation of the gene set in the queried condition; negative scores indicate "
            "down-regulation.\n"
        )
        lines = []
        for gs, val in zip(gene_sets, scores):
            gs_name = gs.replace("gs_", "")
            genes = self.get_genes_for_set(gs_name)
            direction = "up-regulated" if val > 0 else "down-regulated" if val < 0 else "not changed"
            lines.append(f"{gs} has gene set {genes} : {direction} (VISION score = {val:.3f})")

        return header + "\n".join(lines)
    
    def obtain_moa(self, drug_name: str):
        row = self.tahoe_drug_meta[self.tahoe_drug_meta["drug"] == drug_name]

        if row.empty:
            return "MOA annotation not found for this drug."
        
        moa_broad = row["moa-broad"].values[0]
        moa_fine  = row["moa-fine"].values[0]

        return (
            f"Broad MOA: {moa_broad}; "
            f"Fine MOA: {moa_fine}. "
            "Fine-grained mechanism of action (MOA) annotation for the drug, "
            "specifying the biological process or molecular target affected. "
            "Derived from MedChemExpress and curated with GPT-based annotations."
        )

    def obtain_gene_targets(self, drug_name: str):
        row = self.tahoe_drug_meta[self.tahoe_drug_meta["drug"] == drug_name]
        if row.empty:
            return "Gene targets not found for this drug."
        
        targets = row["targets"].values[0]

        # Convert a stringified list/dict to a Python object, if necessary.
        if isinstance(targets, str):
            try:
                targets = eval(targets)
            except Exception:  # fall back to treating it as a single ID
                targets = [targets]

        return (
            f"Gene target token IDs: {targets}. "
            "Gene identifiers (integer token IDs) corresponding to each gene with non-zero expression in the cell."
        )

    def obtain_cell_line_data(self, cell_line_name: str):
        row = self.tahoe_cell_meta[self.tahoe_cell_meta["cell_name"] == cell_line_name]

        if row.empty:
            return "Cell-line metadata not found for this cell line."
        
        organ                 = row["Organ"].values[0]
        driver_gene_symbol    = row["Driver_Gene_Symbol"].values[0]
        driver_varzyg         = row["Driver_VarZyg"].values[0]
        driver_vartype        = row["Driver_VarType"].values[0]
        driver_proteffect     = row["Driver_ProtEffect_or_CdnaEffect"].values[0]
        driver_mech_inferdm   = row["Driver_Mech_InferDM"].values[0]
        driver_genetype_dm    = row["Driver_GeneType_DM"].values[0]

        return (
            f"Organ: {organ}; "
            f"Driver_Gene_Symbol: {driver_gene_symbol}; "
            f"Driver_VarZyg: {driver_varzyg}; "
            f"Driver_VarType: {driver_vartype}; "
            f"Driver_ProtEffect_or_CdnaEffect: {driver_proteffect}; "
            f"Driver_Mech_InferDM: {driver_mech_inferdm}; "
            f"Driver_GeneType_DM: {driver_genetype_dm}. "
            "Organ = tissue or organ of origin for the cell line (e.g., Lung), used to interpret lineage-specific responses. "
            "Driver_Gene_Symbol = HGNC-approved symbol of a driver gene with functional alterations in this cell line. "
            "Driver_VarZyg = zygosity of the driver variant (Hom = homozygous, Het = heterozygous). "
            "Driver_VarType = type of genetic alteration (e.g., Missense, Frameshift, Stopgain). "
            "Driver_ProtEffect_or_CdnaEffect = precise protein or cDNA-level annotation of the mutation (e.g., p.G12S). "
            "Driver_Mech_InferDM = inferred functional mechanism (LoF = loss-of-function, GoF = gain-of-function). "
            "Driver_GeneType_DM = classification of the driver gene as an Oncogene or Suppressor."
        )   

    def run_gradio_chat(self, message: str,
                    history: list,
                    temperature: float,
                    max_new_tokens: int,
                    max_token: int,
                    call_agent: bool,
                    conversation: gr.State,
                    max_round: int = 20,
                    seed: int = None,
                    call_agent_level: int = 0,
                    sub_agent_task: str = None):
    
        print("\033[1;32;40mstart\033[0m")
        print("len(message)", len(message))

        if len(message) <= 10:
            yield "Hi, I am Agent, an assistant for answering biomedical questions. Please provide a valid message with a string longer than 10 characters."
            return "Please provide a valid message."
        
        outputs = []
        outputs_str = ''
        last_outputs = []

        conversation = self.initialize_conversation(
            message,
            conversation=conversation,
            history=history)
        
        history = []

        next_round = True
        function_call_messages = []
        current_round = 0
        enable_summary = False
        last_status = {}  # for summary
        token_overflow = False
        # if self.enable_checker:
        #     checker = ReasoningTraceChecker(
        #         message, conversation, init_index=len(conversation))

        # try:
        self.conversation.append({"role": "user", "content": message})
        while next_round and current_round < max_round:
            current_round += 1

            response = self.llm_infer(self.conversation)
            self.conversation.append({"role": "system", "content": response})
            tool_called = False 
            print(response)
            # import pdb; pdb.set_trace()

            if 'Tool-call:' in response:
                match = re.search(r"Tool-call:\s*(.*)", response, re.DOTALL)
                response_text = match.group(1).strip()
                if "None" not in response_text and response_text.replace('-', '').rstrip().replace('FINISHED', '').rstrip():   
                    history.append(ChatMessage(
                        role="assistant", content=f"{response.replace('FINISHED', '').split('</think>')[1]}"))
                    yield history 
                    
                    tool_called = True
                    print(response_text)
                    if "FAIL" in response_text:
                        self.conversation.append({"role": "system", "content": tool_response})
                        history.append(
                            ChatMessage(role="assistant", content=f"Response from tool FAILED ")
                        )
                        next_round = False
                        yield history 
                    else:        
                        tool_call_text = response_text
                        if ';' in tool_call_text:
                            tool_calls = [i.replace('\n', '').rstrip('-').replace('FINISHED', '') for i in tool_call_text.split(';') if i]
                        elif '\n' in tool_call_text:
                            tool_calls = [i.replace('\n', '').rstrip('-').replace('FINISHED', '') for i in tool_call_text.split('\n') if i]
                        else:
                            tool_calls = [tool_call_text]
                    
                        tool_calls = [i.rstrip('-') for i in tool_calls if i]

                        for call in tool_calls:
                            print(f"\033[1;34;40mCalling this command now {call}\033[0m")
                            tool_response = str(eval(call))
                            self.conversation.append({"role": "system", "content": tool_response})
                            history.append(
                                ChatMessage(role="assistant", content=f"Response from tool: {tool_response}")
                            )
                            print(f"\033[1;34;40mGot this response {tool_response}\033[0m")
                            yield history
                else:
                    history.append(
                                ChatMessage(role="assistant", content=f"{response}")
                            )
                    yield history

            elif 'Response:' in response or tool_called is False:
                match = re.search(r"Response:\s*(.*)", response, re.DOTALL)
                response_text = match.group(1).strip().replace('Tool-call: None', '')
                print(f"\033[1;33;40mresponse text: {response_text}\033[0m")
                history.append(
                    ChatMessage(
                        role="assistant", content=f"{response_text.replace('FINISHED', '')}")
                )
                yield history
                
            if 'FINISHED' in response and tool_called is False:
                next_round = False





      
        #         if len(last_outputs) > 0:
        #             function_call_messages, picked_tools_prompt, special_tool_call, current_gradio_history = yield from self.run_function_call_stream(
        #                 last_outputs, return_message=True,
        #                 existing_tools_prompt=picked_tools_prompt,
        #                 message_for_call_agent=message,
        #                 call_agent=call_agent,
        #                 call_agent_level=call_agent_level,
        #                 temperature=temperature)
        #             history.extend(current_gradio_history)
        #             if special_tool_call == 'Finish':
        #                 yield history
        #                 next_round = False
        #                 conversation.extend(function_call_messages)
        #                 return function_call_messages[0]['content']
        #             elif special_tool_call == 'RequireClarification' or special_tool_call == 'DirectResponse':
        #                 history.append(
        #                     ChatMessage(role="assistant", content=history[-1].content))
        #                 yield history
        #                 next_round = False
        #                 return history[-1].content
        #             if (self.enable_summary or token_overflow) and not call_agent:
        #                 if token_overflow:
        #                     print("token_overflow, using summary")
        #                 enable_summary = True
        #             last_status = self.function_result_summary(
        #                 conversation, status=last_status,
        #                 enable_summary=enable_summary)
        #             if function_call_messages is not None:
        #                 conversation.extend(function_call_messages)
        #                 formated_md_function_call_messages = tool_result_format(
        #                     function_call_messages)
        #                 yield history
        #             else:
        #                 next_round = False
        #                 conversation.extend(
        #                     [{"role": "assistant", "content": ''.join(last_outputs)}])
        #                 return ''.join(last_outputs).replace("</s>", "")
        #         # if self.enable_checker:
        #         #     good_status, wrong_info = checker.check_conversation()
        #         #     if not good_status:
        #         #         next_round = False
        #         #         print("Internal error in reasoning: " + wrong_info)
        #         #         break
        #         last_outputs = []
        #         last_outputs_str, token_overflow = self.llm_infer(
        #             messages=conversation,
        #             temperature=temperature,
        #             tools=picked_tools_prompt,
        #             skip_special_tokens=False,
        #             max_new_tokens=max_new_tokens,
        #             max_token=max_token,
        #             seed=seed,
        #             check_token_status=True)
        #         last_thought = last_outputs_str.split("[TOOL_CALLS]")[0]
        #         for each in history:
        #             if each.metadata is not None:
        #                 each.metadata['status'] = 'done'
        #         if '[FinalAnswer]' in last_thought:
        #             final_thought, final_answer = last_thought.split(
        #                 '[FinalAnswer]')
        #             history.append(
        #                 ChatMessage(role="assistant",
        #                             content=final_thought.strip())
        #             )
        #             yield history
        #             history.append(
        #                 ChatMessage(
        #                     role="assistant", content="**Answer**:\n"+final_answer.strip())
        #             )
        #             yield history
        #         else:
        #             history.append(ChatMessage(
        #                 role="assistant", content=last_thought))
        #             yield history

        #         last_outputs.append(last_outputs_str)

        #     if self.force_finish:
        #         last_outputs_str = self.get_answer_based_on_unfinished_reasoning(
        #             conversation, temperature, max_new_tokens, max_token, return_full_thought=True)
        #         for each in history:
        #             if each.metadata is not None:
        #                 each.metadata['status'] = 'done'

        #         final_thought, final_answer = last_outputs_str.split('[FinalAnswer]')
        #         history.append(
        #             ChatMessage(role="assistant",
        #                         content=final_thought.strip())
        #         )
        #         yield history
        #         history.append(
        #             ChatMessage(
        #                 role="assistant", content="**Answer**:\n"+final_answer.strip())
        #         )
        #         yield history
        #     else:
        #         yield "The number of rounds exceeds the maximum limit!"

        # except Exception as e:
        #     print(f"Error: {e}")
        #     if self.force_finish:
        #         last_outputs_str = self.get_answer_based_on_unfinished_reasoning(
        #             conversation,
        #             temperature,
        #             max_new_tokens,
        #             max_token,
        #             return_full_thought=True)
        #         for each in history:
        #             if each.metadata is not None:
        #                 each.metadata['status'] = 'done'

        #         final_thought, final_answer = last_outputs_str.split(
        #             '[FinalAnswer]')
        #         history.append(
        #             ChatMessage(role="assistant",
        #                         content=final_thought.strip())
        #         )
        #         yield history
        #         history.append(
        #             ChatMessage(
        #                 role="assistant", content="**Answer**:\n"+final_answer.strip())
        #         )
        #         yield history
        #     else:
        #         return None  