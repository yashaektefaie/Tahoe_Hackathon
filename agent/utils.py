from openai import AzureOpenAI, OpenAI
import yaml


class Basic_Agent():

    def __init__(self, config):
        self.config = self.load_config(config)
        self.openai_api_key = self.config['openai_api_key']
        if 'open_api_base' in self.config:
            self.open_api_base = self.config['open_api_base']
        self.azure_openai_api_key = self.config['azure_openai_api_key']
        self.azure_openai_endpoint = self.config['azure_openai_endpoint']
        self.openai_backend = self.config['openai_backend']
        # self.pqapi_token = self.config['pqapi_token']
        # os.environ['PQA_API_TOKEN'] = self.pqapi_token

    def load_config(self,config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

    def llm_infer(self, conversation, temp = 0.000000001, max_tokens = 1000, image = None, role = None):

        while True:

            if self.openai_backend == 'azure':
                client = AzureOpenAI(
                        azure_endpoint = self.azure_openai_endpoint,
                        api_key=self.azure_openai_api_key,
                        api_version="2024-05-01-preview")

                response = client.chat.completions.create(
                        model='gpt-4o',
                        messages = conversation,
                        temperature=temp,
                        max_tokens=max_tokens,
                )
            elif self.openai_backend == 'openai':
                client = OpenAI(
                    api_key=self.openai_api_key
                )

                response = client.chat.completions.create(
                    model='gpt-4o',
                    messages=conversation,
                    temperature=temp,
                    max_tokens=max_tokens,
                )
            elif self.openai_backend == 'lambda':

                client = OpenAI(api_key = self.openai_api_key,
                                base_url = self.open_api_base)
                
                model = "llama-4-maverick-17b-128e-instruct-fp8"
                response = client.chat.completions.create(
                    model = model,
                    messages = conversation)
            else:
                raise ValueError(f"Invalid openai_backend: {self.openai_backend}")

            if "I'm sorry, I can't assist with that" in response.choices[0].message.content or "I'm unable to view the image" in response.choices[0].message.content or "I'm unable to provide a definitive answer" in response.choices[0].message.content:
                    print("Failed to generate response, trying again")
                    continue
            else:
                    response = response.choices[0].message.content
                    return response
        
    def run_function(self, output):
        try:
            tool_call = output.split('Tool-call:')[-1].rstrip().replace('\n', '')
            res = eval(tool_call)
            return res
        except Exception as e:
            print(f"Error in parsing tool call in {output} got this error {e}")
            import pdb; pdb.set_trace()
