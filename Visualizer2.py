import torch
import torch.nn.functional as F
import time
import pygame
import numpy as np

class MLPVisualizer:
    def __init__(self, width=1920, height=1080):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MLP Neural Network Visualizer")
        self.font = pygame.font.Font(None, 24)
        self.colors = {
            'background': (0, 0, 50),    # Dark blue background
            'node': (0, 150, 255),        # Bright cyan-blue for nodes
            'active_node': (0, 255, 0),   # Bright green for active nodes
            'connection': (200, 200, 200),# Light gray for inactive connections
            'text': (255, 255, 255),      # White text
            'button': (200, 200, 200)     # Light gray button
        }
        self.button_rect = pygame.Rect(self.width - 150, self.height - 50, 100, 30)
        self.step_mode = True  # Enable step-by-step mode by default
        self.ready_for_next = True  # Flag to control step progression
        self.current_sample = ""  # Store the current sample
        self.disabled_neurons = {
            'input': set(),
            'embedding': set(),
            'hidden1': set(),  # Renamed from 'hidden' to 'hidden1'
            'hidden2': set(),  # Added new hidden layer
            'output': set()
        }
        self.node_radius = 5  # Store radius as class variable
        self.current_context = None
        self.current_emb = None
        self.current_h1 = None  # First hidden layer
        self.current_h2 = None  # Second hidden layer
        self.current_logits = None
        self.itos = None

    def draw_network(self, C, W1, b1, W2, b2, W3, b3, current_context=None, current_emb=None, 
                    current_h1=None, current_h2=None, current_logits=None, itos=None):
        # Store current state
        self.C = C
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.W3 = W3  # New weight matrix
        self.b3 = b3  # New bias vector
        self.current_context = current_context
        self.current_emb = current_emb
        self.current_h1 = current_h1  # First hidden layer
        self.current_h2 = current_h2  # Second hidden layer
        self.current_logits = current_logits
        self.itos = itos

        self.screen.fill(self.colors['background'])
        
        # Store weights for connection coloring
        self.layer_x = [self.width * (i + 1)/(6) for i in range(5)]  # Adjusted for 5 layers
        
        # Draw current sample at the top of the screen
        sample_text = self.font.render(f"Current Sample: '{self.current_sample}'", True, self.colors['text'])
        self.screen.blit(sample_text, (20, 20))
        
        # Calculate layer positions
        embedding_dim = C.shape[1]  # Dimension of each embedding vector
        if current_context is not None:
            context_size = len(current_context)  # Use actual context length
        else:
            context_size = len(current_emb.view(-1)) // embedding_dim if current_emb is not None else 8
        
        layers = [
            context_size,       # Input nodes (context window size)
            embedding_dim,      # Embedding dimensions
            W1.shape[1],        # First hidden layer size
            W2.shape[1],        # Second hidden layer size
            W3.shape[1]         # Output layer size
        ]
        
        layer_x = [self.width * (i + 1)/(len(layers) + 1) for i in range(len(layers))]
        
        # Draw connections between layers
        self._draw_connections(layer_x[0], layer_x[1], layers[0], layers[1], 'input', 'embedding')
        self._draw_connections(layer_x[1], layer_x[2], layers[1], layers[2], 'embedding', 'hidden1')
        self._draw_connections(layer_x[2], layer_x[3], layers[2], layers[3], 'hidden1', 'hidden2')
        self._draw_connections(layer_x[3], layer_x[4], layers[3], layers[4], 'hidden2', 'output')

        # Draw nodes for each layer with character labels
        node_positions = []
        layer_names = ['input', 'embedding', 'hidden1', 'hidden2', 'output']
        for i, (x, nodes, name) in enumerate(zip(layer_x, layers, layer_names)):
            positions = self._draw_layer(x, nodes, name.capitalize(), 
                                      input_chars=current_context if i == 0 and current_context else None,
                                      output_chars=itos if i == 4 else None,  # Output layer is now index 4
                                      layer_name=name)
            node_positions.append(positions)

        self.current_positions = node_positions  # Store positions for click detection

        # Highlight active nodes if provided
        if current_emb is not None:
            emb_reshaped = current_emb.view(-1)
            self._highlight_active_nodes(node_positions[1], emb_reshaped, show_values=True, layer_name='embedding')
        if current_h1 is not None:
            self._highlight_active_nodes(node_positions[2], current_h1, show_values=True, layer_name='hidden1')
        if current_h2 is not None:
            self._highlight_active_nodes(node_positions[3], current_h2, show_values=True, layer_name='hidden2')
        if current_logits is not None:
            self._highlight_active_nodes(node_positions[4], current_logits, show_values=False, layer_name='output')

        # Draw step button
        pygame.draw.rect(self.screen, self.colors['button'], self.button_rect)
        text = self.font.render('Next Step', True, self.colors['text'])
        text_rect = text.get_rect(center=self.button_rect.center)
        self.screen.blit(text, text_rect)

        pygame.display.flip()

    def _draw_connections(self, x1, x2, nodes1, nodes2, layer1_name, layer2_name):
        y1_step = self.height * 0.8 / (nodes1 + 1)
        y2_step = self.height * 0.8 / (nodes2 + 1)
        y1_start = self.height * 0.1
        y2_start = self.height * 0.1
        
        # Get the weights between these layers
        if layer1_name == 'input' and layer2_name == 'embedding':  # Input to Embedding
            weights = self.C.detach().cpu().numpy()
        elif layer1_name == 'embedding' and layer2_name == 'hidden1':  # Embedding to Hidden1
            weights = self.W1.detach().cpu().numpy()
        elif layer1_name == 'hidden1' and layer2_name == 'hidden2':  # Hidden1 to Hidden2
            weights = self.W2.detach().cpu().numpy()
        else:  # Hidden2 to Output
            weights = self.W3.detach().cpu().numpy()
        
        # Normalize weights to [0, 1] range
        max_weight = np.max(np.abs(weights))
        if max_weight != 0:
            normalized_weights = weights / max_weight
        else:
            normalized_weights = weights
        
        for i in range(nodes1):
            if (layer1_name, i) in self.disabled_neurons[layer1_name]:
                continue
            
            for j in range(nodes2):
                if (layer2_name, j) in self.disabled_neurons[layer2_name]:
                    continue
                
                start_pos = (x1, y1_start + (i + 1) * y1_step)
                end_pos = (x2, y2_start + (j + 1) * y2_step)
                
                # Get the weight value and determine color
                weight = normalized_weights[i % weights.shape[0], j % weights.shape[1]]
                intensity = int(min(max(abs(weight) * 255, 0), 255))
                
                if weight > 0:
                    # Bright neon green for positive weights
                    color = (0, intensity, 20)
                else:
                    # Vibrant red for negative weights
                    color = (intensity, 0, 20)
                
                pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

    def _draw_layer(self, x, num_nodes, label, input_chars=None, output_chars=None, layer_name=None):
        positions = []
        y_step = self.height * 0.8 / (num_nodes + 1)
        y_start = self.height * 0.1
        
        # Draw label
        text = self.font.render(label, True, self.colors['text'])
        self.screen.blit(text, (x - text.get_width()/2, y_start - 30))
        
        for i in range(num_nodes):
            y = y_start + (i + 1) * y_step
            color = self.colors['connection'] if (layer_name, i) in self.disabled_neurons[layer_name] else self.colors['node']
            pygame.draw.circle(self.screen, color, (x, y), self.node_radius)
            positions.append((x, y))
            
            # Draw character labels
            if input_chars is not None and i < len(input_chars):
                char_text = self.font.render(input_chars[i], True, self.colors['text'])
                self.screen.blit(char_text, (x - 30, y - 8))
            elif output_chars is not None and i < len(output_chars):
                char_text = self.font.render(output_chars[i], True, self.colors['text'])
                self.screen.blit(char_text, (x + 15, y - 8))
                
        return positions

    def _highlight_active_nodes(self, positions, values, show_values=True, layer_name=None):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = values.flatten()
        
        # Handle the normalization more safely
        abs_max = np.max(np.abs(values))
        if abs_max == 0:
            normalized = np.zeros_like(values)
        else:
            normalized = values / abs_max  # Normalize to [-1, 1] range
        
        normalized = np.nan_to_num(normalized, nan=0.0)
        
        for pos, norm_value, orig_value in zip(positions, normalized, values):
            intensity = int(255 * abs(norm_value))
            
            # Use red for negative, green for positive activations
            if orig_value > 0:
                color = (0, intensity, 0)  # Green for positive
            else:
                color = (intensity, 0, 0)  # Red for negative
            
            # Draw outer glow
            glow_radius = self.node_radius + 3
            glow_color = (color[0]//3, color[1]//3, color[2]//3)
            pygame.draw.circle(self.screen, glow_color, pos, glow_radius)
            
            # Draw the active node
            pygame.draw.circle(self.screen, color, pos, self.node_radius)
            
            # Draw activation values and biases for hidden layers
            if show_values:
                if layer_name in ['hidden1', 'hidden2']:
                    # Get the corresponding bias value
                    bias_idx = positions.index(pos)
                    if layer_name == 'hidden1':
                        bias_value = self.b1[bias_idx].item()
                    else:
                        bias_value = self.b2[bias_idx].item()
                    # Show both activation and bias
                    value_text = f"a:{orig_value:.2f}\nb:{bias_value:.2f}"
                else:
                    value_text = f"{orig_value:.2f}"
                
                # Split text into lines and render
                lines = value_text.split('\n')
                for i, line in enumerate(lines):
                    text = self.font.render(line, True, self.colors['text'])
                    text_pos = (pos[0] - text.get_width()/2, pos[1] - 45 + i*20)
                    self.screen.blit(text, text_pos)

    def _is_node_clicked(self, mouse_pos, node_pos):
        """Check if a node was clicked"""
        return ((mouse_pos[0] - node_pos[0])**2 + 
                (mouse_pos[1] - node_pos[1])**2 <= (self.node_radius * 2)**2)

    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.button_rect.collidepoint(event.pos):
                    self.ready_for_next = True
                    return True
                else:
                    # Check if any node was clicked
                    for layer_name, positions in zip(['input', 'embedding', 'hidden1', 'hidden2', 'output'], 
                                                  self.current_positions):
                        for i, pos in enumerate(positions):
                            if self._is_node_clicked(event.pos, pos):
                                # Toggle neuron state
                                if (layer_name, i) in self.disabled_neurons[layer_name]:
                                    self.disabled_neurons[layer_name].remove((layer_name, i))
                                else:
                                    self.disabled_neurons[layer_name].add((layer_name, i))
                                # Redraw the network with current state
                                self.draw_network(self.C, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3,
                                               current_context=self.current_context,
                                               current_emb=self.current_emb,
                                               current_h1=self.current_h1,
                                               current_h2=self.current_h2,
                                               current_logits=self.current_logits,
                                               itos=self.itos)
                                return True
        return True

    def wait_for_step(self):
        if not self.step_mode:
            return True
        
        self.ready_for_next = False
        while not self.ready_for_next:
            if not self.check_events():
                return False
            pygame.time.wait(50)
        return True

def load_model_and_sample(filename, num_samples=20, seed=None):
    # Load the model - updated to handle two hidden layers
    checkpoint = torch.load(filename)
    C = checkpoint['C']
    W1 = checkpoint['W1']
    b1 = checkpoint['b1']
    W2 = checkpoint['W2']
    b2 = checkpoint['b2']
    W3 = checkpoint['W3']  # New weight matrix
    b3 = checkpoint['b3']  # New bias vector
    itos = checkpoint['itos']
    context_size = checkpoint['context_size']
    vocab_size = checkpoint['vocab_size']
    
    device = C.device
    embedding_dim = C.shape[1]
    hidden_size1 = W1.shape[1]
    hidden_size2 = W2.shape[1]
    
    # Print dimensions for debugging
    print(f"Model dimensions:")
    print(f"C shape: {C.shape} (vocab_size, embedding_dim)")
    print(f"W1 shape: {W1.shape}")
    print(f"W2 shape: {W2.shape}")
    print(f"W3 shape: {W3.shape}")
    print(f"Block size: {context_size}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Hidden size 1: {hidden_size1}")
    print(f"Hidden size 2: {hidden_size2}")
    
    # Reshape W1 to handle concatenated embeddings
    W1 = W1.expand(context_size * embedding_dim, hidden_size1)
    
    print(f"W1 shape after reshape: {W1.shape}")
    
    visualizer = MLPVisualizer()
    
    g = torch.Generator(device = device).manual_seed(2147483647) # for reproducibility
    
    print(f"Generating {num_samples} samples:")
    for i in range(num_samples):
        out = []
        context = [0] * context_size
        visualizer.current_sample = ""
        
        while True:
            # Convert context indices to characters for display
            context_chars = [itos[idx] for idx in context]
            
            # Get embeddings and reshape properly
            emb = C[torch.tensor(context, device=device)]  # Shape: [context_size, emb_dim]
            emb_flat = emb.view(-1).unsqueeze(0)  # Reshape to [1, context_size * emb_dim]

            # Forward pass - updated for two hidden layers
            h1 = torch.tanh(emb_flat @ W1 + b1)  # First hidden layer
            h2 = torch.tanh(h1 @ W2 + b2)        # Second hidden layer
            logits = h2 @ W3 + b3                # Output layer
            
            # Visualize the current state - updated for two hidden layers
            visualizer.draw_network(C, W1, b1, W2, b2, W3, b3,
                                 current_context=context_chars,
                                 current_emb=emb,
                                 current_h1=h1,
                                 current_h2=h2,
                                 current_logits=logits,
                                 itos=itos)
            
            # Wait for user to click "Next Step"
            if not visualizer.wait_for_step():
                pygame.quit()
                return
            
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            if ix != 0:
                out.append(ix)
                visualizer.current_sample += itos[ix]  # Add new character to current sample
            if ix == 0:
                break
        
        print(f"{i+1}. {''.join(itos[i] for i in out)}")
    
    pygame.quit()
    # Return the loaded model components in case they're needed
    return {
        'C': C, 
        'W1': W1, 
        'b1': b1, 
        'W2': W2, 
        'b2': b2,
        'W3': W3,
        'b3': b3,
        'itos': itos,
        'context_size': context_size,
        'vocab_size': vocab_size
    }
    
# Example usage:    
load_model_and_sample("MLP2.w", 20, 42)