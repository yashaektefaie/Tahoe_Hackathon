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
                print(f"FAIL: IC50 value is missing for '{drug_name}' in cell line '{cell_line_name}' (depmap_id: {depmap_id}).")
                return f"FAIL: IC50 value is missing for '{drug_name}' in cell line '{cell_line_name}' (depmap_id: {depmap_id})."
            return float(ic50_val)
        except KeyError as e:
            print(f"Combination not found: {e}")
            return None
    
    def rank_vision_scores(self, drug_name: str, cell_line_name: str, k_value: int):
        self.tahoe_vision_scores.X = (self.tahoe_vision_scores.X - np.mean(self.tahoe_vision_scores.X, axis = 0)) / np.std(self.tahoe_vision_scores.X, axis = 0)

        filtered_scores = self.tahoe_vision_scores[
            (self.tahoe_vision_scores.obs["Cell_Name_Vevo"] == cell_line_name) & 
            (self.tahoe_vision_scores.obs["drug"] == drug_name)
        ]

        filtered_scores = filtered_scores[filtered_scores.obs["concentration"] == filtered_scores.obs["concentration"].max()]

        sorted_indices = np.argsort(-np.abs(filtered_scores.X[0]))[:k_value]
        filtered_scores = filtered_scores[:, sorted_indices]
        
        gene_set_names = filtered_scores.var.index.tolist()
        vision_values = [str(value) for value in filtered_scores.X[0].tolist()]

        return list(zip(gene_set_names, vision_values))
    
    def obtain_moa(self, drug_name: str):
        row = self.tahoe_drug_meta[self.tahoe_drug_meta["drug"] == drug_name]

        if row.empty:
            return None
        return {
            "moa-broad": row["moa-broad"].values[0],
            "moa-fine": row["moa-fine"].values[0]
        }

    def obtain_gene_targets(self, drug_name: str):
        row = self.tahoe_drug_meta[self.tahoe_drug_meta["drug"] == drug_name]
        if row.empty:
            return None
        
        targets = row["targets"].values[0]

        if isinstance(targets, str):
            try:
                targets = eval(targets)
            except:
                targets = [targets]

        return targets

    def obtain_cell_line_data(self, cell_line_name: str):
        row = self.tahoe_cell_meta[self.tahoe_cell_meta["cell_name"] == cell_line_name]

        if row.empty:
            return None
        
        return {
            "organ": row["Organ"].values[0],
            "driver_gene_symbol": row["Driver_Gene_Symbol"],
            "driver_varzyg": row["Driver_VarZyg"].values[0],
            "driver_vartype": row["Driver_VarType"].values[0],
            "driver_proteffect_cdnaeffect": row["Driver_ProtEffect_or_CdnaEffect"].values[0],
            "driver_mech_inferdm": row["Driver_Mech_InferDM"].values[0],
            "driver_genetype_dm": row["Driver_GeneType_DM"].values[0]
        }   

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

            if 'Tool-call:' in response:
                match = re.search(r"Tool-call:\s*(.*)", response, re.DOTALL)
                response_text = match.group(1).strip()
                if "None" not in response_text and response_text.replace('-', '').rstrip().replace('FINISHED', '').rstrip():   
                    history.append(ChatMessage(
                        role="assistant", content=f"{response.replace('FINISHED', '')}"))
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
                            tool_calls = [i.replace('\n', '').rstrip('-').replace('FINISHED', '') for i in tool_call_text.split(';')]
                        elif '\n' in tool_call_text:
                            tool_calls = [i.replace('\n', '').rstrip('-').replace('FINISHED', '') for i in tool_call_text.split('\n')]
                        else:
                            tool_calls = [tool_call_text]
                    
                        for call in tool_calls:
                            print(f"Calling this command now {call}")
                            tool_response = str(eval(call))
                            self.conversation.append({"role": "system", "content": tool_response})
                            history.append(
                                ChatMessage(role="assistant", content=f"Response from tool: {tool_response}")
                            )
                            print(f"Got this response {tool_response}")
                            yield history
                else:
                    history.append(
                                ChatMessage(role="assistant", content=f"{response}")
                            )
                    yield history

            elif 'Response:' in response or tool_called is False:
                match = re.search(r"Response:\s*(.*)", response, re.DOTALL)
                response_text = match.group(1).strip().replace('Tool-call: None', '')
                print(response_text)
                history.append(
                    ChatMessage(
                        role="assistant", content=f"{response_text.replace('FINISHED', '')}")
                )
                yield history
                
            if 'FINISHED' in response:
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