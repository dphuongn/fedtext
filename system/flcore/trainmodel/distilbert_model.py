import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

from transformers import AutoModelForSequenceClassification


class LoRALayer(nn.Module):
    def __init__(self, 
         in_dim, 
         out_dim, 
         rank: int, 
         alpha: int, 
         dropout: float, 
         # merge_weights: bool
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        # self.merge_weights = merge_weights

        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.W_a = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.W_b = nn.Parameter(torch.zeros(rank, out_dim))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else (lambda x: x)

        # Scaling
        # self.scaling = self.alpha / self.rank
        self.scaling = self.alpha

        # Merged flag
        # self.merged = False

    def forward(self, x):
        # if self.rank > 0 and not self.merged:
        if self.rank > 0:
            x = self.dropout(x)
            x = self.scaling * (x @ self.W_a @ self.W_b)
        return x

    # def merge(self):
    #     if self.merge_weights and not self.merged:
    #         self.W_b.data += self.W_a @ self.W_b * self.scaling
    #         self.merged = True
            
class LinearWithLoRA(nn.Module):
    def __init__(self, 
         linear, 
         rank: int = 0, 
         alpha: int = 1, 
         dropout: float = 0.0, 
         # merge_weights: bool = True,
    ):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, 
            linear.out_features, 
            rank, 
            alpha, 
            dropout, 
            # merge_weights
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

    # def merge(self):
    #     self.lora.merge()


# class LoRALayer(torch.nn.Module):
#     def __init__(self, in_dim, out_dim, rank, alpha):
#         super().__init__()
#         std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
#         self.W_a = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
#         self.W_b = torch.nn.Parameter(torch.zeros(rank, out_dim))
#         self.alpha = alpha

#     def forward(self, x):
#         x = self.alpha * (x @ self.W_a @ self.W_b)
#         return x


# class LinearWithLoRA(torch.nn.Module):
#     def __init__(self, linear, rank, alpha):
#         super().__init__()
#         self.linear = linear
#         self.lora = LoRALayer(
#             linear.in_features, linear.out_features, rank, alpha
#         )

#     def forward(self, x):
#         return self.linear(x) + self.lora(x)

    

class DistilBertModelWithLoRA(torch.nn.Module):
    def __init__(self, model_id, home_dir, lora_params, num_labels):
        """
        Initialize the DistilBert model with LoRA layers.
        
        Args:
            model_id (str): Identifier for the pre-trained DistilBert model.
            home_dir (str): Directory path for model and processor caching.
            lora_params (dict): Parameters for configuring the LoRA layers.
        """
        super().__init__()
        self.model_id = model_id
        self.home_dir = home_dir
        self.lora_params = lora_params
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=self.num_labels)
        
        org_num_params = self.count_parameters()
        
        # print(f'original distilbert num params: {org_num_params:,}')
        
        org_model_size = self.calculate_model_size(self.model)
        
        # print(f'original distilbert model size: {org_model_size:.3f} MB')
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        self._apply_lora()
        
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def count_all_parameters(self):
        return sum(p.numel() for p in self.model.parameters())
    
    def calculate_model_size(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        
        return size_all_mb
    
    def calculate_state_dict_size(self, state_dict):
        total_size = 0
        for name, tensor in state_dict.items():
            total_size += tensor.nelement() * tensor.element_size()

        size_all_mb = total_size / 1024 ** 2  # Convert to megabytes
        return size_all_mb

        
    def _apply_lora(self):
        """
        Apply LoRA modifications to the DistilBert model. This method initializes
        LoRA layers and replaces the corresponding layers in the DistilBert model.
        """
        assign_lora = partial(
            LinearWithLoRA,
            rank=self.lora_params['rank'],
            alpha=self.lora_params['alpha'],
            dropout=self.lora_params['dropout'],
        )
        
        # Initialize a dictionary to keep track of the LoRA layers
        self.lora_layers = {}
        
        for layer in self.model.distilbert.transformer.layer:
            if self.lora_params['lora_query']:
                layer.attention.q_lin = assign_lora(layer.attention.q_lin)
            if self.lora_params['lora_key']:
                layer.attention.k_lin = assign_lora(layer.attention.k_lin)
            if self.lora_params['lora_value']:
                layer.attention.v_lin = assign_lora(layer.attention.v_lin)
            if self.lora_params['lora_projection']:
                layer.attention.out_lin = assign_lora(layer.attention.out_lin)
            if self.lora_params['lora_mlp']:
                layer.ffn.lin1 = assign_lora(layer.ffn.lin1)
                layer.ffn.lin2 = assign_lora(layer.ffn.lin2)
        if self.lora_params['lora_head']:
            self.model.pre_classifier = assign_lora(self.model.pre_classifier)
            self.model.classifier = assign_lora(self.model.classifier)
            
            
    def _find_lora_layers(self, module, lora_params):
        """
        Recursively find all LoRA layers in the model.

        Args:
            module (torch.nn.Module): The module (or sub-module) to search within.
            lora_params (list): A list to append the parameters of LoRA layers to.
        """
        for child in module.children():
            if isinstance(child, LinearWithLoRA):
                # Assuming LoRALayer is a component of LinearWithLoRA
                lora_params.extend(list(child.lora.parameters()))
            elif isinstance(child, LoRALayer):
                # Directly collecting parameters from LoRALayer
                lora_params.extend(list(child.parameters()))
            else:
                self._find_lora_layers(child, lora_params)

    def get_lora_parameters(self):
        """
        Retrieve all parameters from the LoRA layers in the model.

        Returns:
            list: A list of parameters from all the LoRA layers.
        """
        lora_params = []
        self._find_lora_layers(self.model, lora_params)
        return lora_params
        
    def save_lora_state_dict(self, directory, filename):
        """
        Save the state dictionary of the LoRA layers to a specified file in a given directory.

        Args:
            directory (str): The directory where the state dict file will be saved.
            filename (str): The name of the file to save the state dict.
        """
        lora_params = self.get_lora_parameters()
        state_dict = {f'lora_param_{i}': param.data for i, param in enumerate(lora_params)}

        # Ensure the directory exists, create if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, filename)
        torch.save(state_dict, file_path)
        
    
    def get_lora_state_dict(self):
        """
        Retrieve the state dictionary of the LoRA layers.

        Returns:
            dict: A state dictionary containing the parameters of the LoRA layers.
        """
        lora_params = self.get_lora_parameters()
        state_dict = {f'lora_param_{i}': param.data for i, param in enumerate(lora_params)}
        return state_dict
    
    def set_lora_state_dict(self, state_dict):
        """
        Set the parameters of the LoRA layers from a state dictionary.

        Args:
            state_dict (dict): A state dictionary containing parameters for the LoRA layers.
        """
        lora_params = self.get_lora_parameters()
        for i, param in enumerate(lora_params):
            param_key = f'lora_param_{i}'
            if param_key in state_dict:
                param.data.copy_(state_dict[param_key])
            
    def update_lora_from_state_dict(self, state_dict):
        """
        Update the parameters of the LoRA layers from a state dictionary.

        Args:
            state_dict (dict): A state dictionary containing the parameters of the LoRA layers.
        """
        lora_params = self.get_lora_parameters()

        for i, param in enumerate(lora_params):
            param_key = f'lora_param_{i}'
            if param_key in state_dict:
                param.data.copy_(state_dict[param_key])
            else:
                raise KeyError(f"Parameter key {param_key} not found in the provided state_dict.")  
                
    def load_lora_state_dict(self, file_path):
        """
        Load LoRA parameters from a saved state dictionary file into the model.

        This method updates the LoRA layers in the model with parameters loaded from a file.

        Args:
            file_path (str): The path to the file containing the saved state dictionary.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            KeyError: If a parameter key in the state dictionary does not match any in the model.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        # Load the state dictionary from the file
        state_dict = torch.load(file_path)

        # Update the model's LoRA parameters with the loaded state dictionary
        self.set_lora_state_dict(state_dict)

                
    def print_dict_shapes(self, dictionary):
        """
        Print the shapes of tensors stored in a dictionary.

        This function iterates over each key-value pair in the dictionary.
        It assumes that each value is a tensor and prints the shape of each tensor
        along with its corresponding key.

        Args:
            dictionary (dict): A dictionary where each value is expected to be a tensor.
        """
        for key, tensor in dictionary.items():
            print(f"Shape of '{key}': {tensor.shape}")
            
    def print_dict_values(self, dictionary):
        """
        Print the values of tensors stored in a dictionary.

        This function iterates over each key-value pair in the dictionary.
        It assumes that each value is a tensor and prints the actual values of each tensor
        along with its corresponding key.

        Note: Be cautious when using this method with large tensors, as printing 
        large amounts of data can be time-consuming and may clutter your output.

        Args:
            dictionary (dict): A dictionary where each value is expected to be a tensor.
        """
        for key, tensor in dictionary.items():
            print(f"{key}:\n{tensor}")
            
    def compare_lora_dictionaries(self, dict1, dict2, tolerance=1e-6):
        """
        Compare two dictionaries containing LoRA parameters.

        Args:
            dict1 (dict): The first dictionary of LoRA parameters.
            dict2 (dict): The second dictionary of LoRA parameters.
            tolerance (float): Tolerance level for comparing floating point values.

        Returns:
            bool: True if the dictionaries are the same within the given tolerance, False otherwise.
        """
        
        if dict1.keys() != dict2.keys():
            return False
        
        for key in dict1:
            if key not in dict2 or not torch.allclose(dict1[key], dict2[key], atol=tolerance):
                return False

        return True
    
    
class BertModelFFT(torch.nn.Module):
    def __init__(self, model_id, home_dir, num_labels):
        """
        Initialize the DistilBert model.
        
        Args:
            model_id (str): Identifier for the pre-trained DistilBert model.
            home_dir (str): Directory path for model and processor caching.
        """
        super().__init__()
        self.model_id = model_id
        self.home_dir = home_dir
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=self.num_labels)
        
    def set_fft_parameters(self, new_model):
        """
        Set the parameters of the new_model to the current model.

        Args:
            new_model (torch.nn.Module): The new model whose parameters will be copied to the current model.
        """
        current_model_dict = self.model.state_dict()
        new_model_dict = new_model.state_dict()
        
        for param_name in current_model_dict:
            if param_name in new_model_dict:
                current_model_dict[param_name].copy_(new_model_dict[param_name])
            else:
                raise KeyError(f"Parameter {param_name} not found in the new model.")
                
                
    
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)

    HOME = '/work/LAS/jannesar-lab/dphuong/jupyter'
    model_id = "distilbert-base-uncased"
    
    
    
    
    lora_params = {
        'rank': 8,
        'alpha': 32,
        'lora_dropout': 0.05,

        'lora_query': True,
        # 'lora_query': False,
        
        'lora_key': True,
        # 'lora_key': False,
        
        'lora_value': True,
        # 'lora_value': False,
        
        'lora_projection': True,
        # 'lora_projection': False,
        
        'lora_mlp': True,
        # 'lora_mlp': False,
        
        'lora_head': True,
        # 'lora_head': False,
    }

    
    DistilBertModelWithLoRA_object1 = DistilBertModelWithLoRA(model_id=model_id, home_dir=HOME, lora_params=lora_params).to(device)
    
    model = DistilBertModelWithLoRA_object1.model
    
    lora_params =  DistilBertModelWithLoRA_object1.count_parameters()
    
    all_params = DistilBertModelWithLoRA_object1.count_all_parameters()
    
    print(f'lora params: {lora_params:,}')
    
    print(f'all params: {all_params:,}')
    
    model_size = DistilBertModelWithLoRA_object1.calculate_model_size(model)
    
    print(f'model size: {model_size:.3f} MB')
    
    lora_state_dict = DistilBertModelWithLoRA_object1.get_lora_state_dict()
    
    lora_size = DistilBertModelWithLoRA_object1.calculate_state_dict_size(lora_state_dict)
    
    print(f'lora size: {lora_size:.3f} MB')
