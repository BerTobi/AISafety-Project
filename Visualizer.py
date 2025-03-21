import torch
import torch.nn.functional as F
import time
import pygame
import numpy as np
import matplotlib.pyplot as plt
import traceback
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import Counter

class MLPVisualizer:
    def __init__(self, width=1920, height=1080):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)  # Make window resizable
        pygame.display.set_caption("MLP Neural Network Visualizer")
        self.font = pygame.font.Font(None, 24)
        self.colors = {
            'background': (0, 0, 50),    # Dark blue background
            'node': (0, 150, 255),           # Bright cyan-blue for nodes
            'active_node': (0, 255, 0),      # Bright green for active nodes
            'connection': (200, 200, 200),    # Light gray for inactive connections
            'text': (255, 255, 255),               # White text
            'button': (200, 200, 200)        # Light gray button
        }
        self.button_rect = pygame.Rect(self.width - 150, self.height - 50, 100, 30)
        self.custom_input_button = pygame.Rect(self.width - 300, self.height - 50, 130, 30)
        self.embedding_button = pygame.Rect(self.width - 450, self.height - 50, 130, 30)
        self.step_mode = True  # Enable step-by-step mode by default
        self.ready_for_next = True  # Flag to control step progression
        self.current_sample = ""  # Add this line to store the current sample
        self.disabled_neurons = {
            'input': set(),
            'embedding': set(),
            'hidden': set(),
            'output': set()
        }
        self.node_radius = 5  # Store radius as class variable
        self.current_context = None
        self.current_emb = None
        self.current_h = None
        self.current_logits = None
        self.itos = None
        self.stoi = None  # Add dictionary to map characters to indices
        self.custom_input = ""  # Store custom input
        self.is_input_active = False  # Flag for text input mode
        self.input_text = ""  # Current text being entered
        self.has_new_custom_input = False  # Flag to indicate new custom input is ready
        self.show_embedding_view = False
        self.dim1 = 0  # First dimension to plot
        self.dim2 = 1  # Second dimension to plot
        self.embedding_surface = None
        self.embedding_plot_size = (800, 800)  # Fixed size for embedding visualization

    def create_stoi(self, itos):
        """Create a mapping from characters to indices"""
        self.stoi = {ch: i for i, ch in enumerate(itos)}
        print(f"Created character mapping with {len(self.stoi)} entries")

    def draw_embedding_view(self, C, itos):
        """Draw 2D embedding visualization using selected dimensions"""
        plt.clf()
        
        try:
            # Create matplotlib figure with fixed size
            fig = plt.figure(figsize=(8, 8), dpi=100, facecolor='#000032')
            ax = fig.add_subplot(111)
            ax.set_facecolor('#000032')
            
            # Get embeddings and plot
            embeddings = C.detach().cpu().numpy()
            x = embeddings[:, self.dim1]
            y = embeddings[:, self.dim2]
            
            # Plot scatter with white dots and labels
            scatter = ax.scatter(x, y, c='white', alpha=0.6, s=100)
            
            # Add character labels with better visibility
            for i, char in enumerate(itos):
                # Escape special characters for display and handle whitespace
                if char.isspace():
                    display_char = '␣'
                elif char == '\n':
                    display_char = '⏎'
                else:
                    display_char = repr(char)[1:-1]
                
                ax.annotate(display_char, 
                           (x[i], y[i]),
                           color='white',
                           fontsize=12,
                           xytext=(5, 5),
                           textcoords='offset points',
                           bbox=dict(facecolor='#000032', edgecolor='white', alpha=0.7))
            
            # Style the plot
            ax.set_title(f'Character Embeddings (dims {self.dim1} vs {self.dim2})',
                        color='white', pad=20, fontsize=14)
            ax.set_xlabel(f'Dimension {self.dim1}', color='white', fontsize=12)
            ax.set_ylabel(f'Dimension {self.dim2}', color='white', fontsize=12)
            
            # Style the axes
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.tick_params(colors='white', labelsize=10)
            ax.grid(True, color='white', alpha=0.2)
            
            # Tight layout to prevent text cutoff
            plt.tight_layout()
            
            # Convert to pygame surface with fixed size
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            
            # Get the RGBA buffer
            buf = canvas.buffer_rgba()
            arr = np.asarray(buf)
            
            # Scale the array to our fixed embedding plot size
            surf = pygame.image.frombuffer(arr.tobytes(), arr.shape[1::-1], "RGBA")
            self.embedding_surface = pygame.transform.scale(surf, self.embedding_plot_size)
            
            plt.close(fig)  # Clean up matplotlib resources
            
        except Exception as e:
            print(f"Error creating embedding visualization: {e}")
            traceback.print_exc()
            self.embedding_surface = None
            self.show_embedding_view = False

    def draw_network(self, C, W1, b1, W2, b2, current_context=None, current_emb=None, current_h=None, current_logits=None, itos=None):
        # Store current state
        self.C = C
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.current_context = current_context
        self.current_emb = current_emb
        self.current_h = current_h
        self.current_logits = current_logits
        self.itos = itos
        
        # Create stoi if it doesn't exist
        if self.stoi is None and itos is not None:
            self.create_stoi(itos)

        self.screen.fill(self.colors['background'])
        
        # Draw text input screen if active
        if self.is_input_active:
            # Create centered input box
            input_rect = pygame.Rect(self.width // 4, self.height // 2 - 50, self.width // 2, 40)
            pygame.draw.rect(self.screen, (50, 50, 80), input_rect)
            pygame.draw.rect(self.screen, (100, 100, 150), input_rect, 2)
            
            # Add prompt text above the input box
            input_prompt = self.font.render("Enter custom text (press RETURN when done):", True, self.colors['text'])
            self.screen.blit(input_prompt, (self.width // 4, self.height // 2 - 80))
            
            # Display the text being typed
            text_surface = self.font.render(self.input_text, True, self.colors['text'])
            self.screen.blit(text_surface, (input_rect.x + 10, input_rect.y + 10))
            
            # Create a "Cancel" button
            cancel_rect = pygame.Rect(self.width // 2 - 50, self.height // 2 + 20, 100, 30)
            pygame.draw.rect(self.screen, self.colors['button'], cancel_rect)
            cancel_text = self.font.render("Cancel", True, (0, 0, 0))
            cancel_text_rect = cancel_text.get_rect(center=cancel_rect.center)
            self.screen.blit(cancel_text, cancel_text_rect)
            
            pygame.display.flip()
            return
        
        # Store weights for connection coloring
        self.layer_x = [self.width * (i + 1)/(5) for i in range(4)]
        
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
            context_size,             # Input nodes (context window size)
            embedding_dim,          # Embedding dimensions
            W1.shape[1],           # Hidden layer size (directly from W1)
            W2.shape[1]            # Output layer size
        ]
        
        layer_x = [self.width * (i + 1)/(len(layers) + 1) for i in range(len(layers))]
        
        # Draw connections with correct node counts and layer names
        self._draw_connections(layer_x[0], layer_x[1], layers[0], layers[1], 'input', 'embedding')
        self._draw_connections(layer_x[1], layer_x[2], layers[1], layers[2], 'embedding', 'hidden')
        self._draw_connections(layer_x[2], layer_x[3], layers[2], layers[3], 'hidden', 'output')

        # Draw nodes for each layer with character labels
        node_positions = []
        layer_names = ['input', 'embedding', 'hidden', 'output']
        for i, (x, nodes, name) in enumerate(zip(layer_x, layers, layer_names)):
            positions = self._draw_layer(x, nodes, name.capitalize(), 
                                      input_chars=current_context if i == 0 and current_context else None,
                                      output_chars=itos if i == 3 else None,
                                      layer_name=name)
            node_positions.append(positions)

        self.current_positions = node_positions  # Store positions for click detection

        # Highlight active nodes if provided
        if current_emb is not None:
            emb_reshaped = current_emb.view(-1)
            self._highlight_active_nodes(node_positions[1], emb_reshaped, show_values=True, layer_name='embedding')
        if current_h is not None:
            self._highlight_active_nodes(node_positions[2], current_h, show_values=True, layer_name='hidden')
        if current_logits is not None:
            self._highlight_active_nodes(node_positions[3], current_logits, show_values=False, layer_name='output')  # Don't show values for output layer

        # Draw step button
        pygame.draw.rect(self.screen, self.colors['button'], self.button_rect)
        text = self.font.render('Next Step', True, (0, 0, 0))
        text_rect = text.get_rect(center=self.button_rect.center)
        self.screen.blit(text, text_rect)
        
        # Draw custom input button
        pygame.draw.rect(self.screen, self.colors['button'], self.custom_input_button)
        custom_text = self.font.render('Custom Input', True, (0, 0, 0))
        custom_text_rect = custom_text.get_rect(center=self.custom_input_button.center)
        self.screen.blit(custom_text, custom_text_rect)

        # Draw embedding view button
        pygame.draw.rect(self.screen, self.colors['button'], self.embedding_button)
        emb_text = self.font.render('Embedding View', True, (0, 0, 0))
        emb_text_rect = emb_text.get_rect(center=self.embedding_button.center)
        self.screen.blit(emb_text, emb_text_rect)
        
        # Draw embedding view if active
        if self.show_embedding_view and self.embedding_surface is not None:
            # Calculate centered position for embedding plot
            plot_rect = self.embedding_surface.get_rect()
            plot_rect.center = (self.width//2, self.height//2)
            
            # Draw semi-transparent background
            bg_surf = pygame.Surface((self.width, self.height))
            bg_surf.fill((0, 0, 50))
            bg_surf.set_alpha(200)
            self.screen.blit(bg_surf, (0, 0))
            
            # Draw the embedding plot
            self.screen.blit(self.embedding_surface, plot_rect)

        pygame.display.flip()

    def _draw_connections(self, x1, x2, nodes1, nodes2, layer1_name, layer2_name):
        y1_step = self.height * 0.8 / (nodes1 + 1)
        y2_step = self.height * 0.8 / (nodes2 + 1)
        y1_start = self.height * 0.1
        y2_start = self.height * 0.1
        
        # Get the weights between these layers
        if x1 == self.layer_x[0]:  # Input to Embedding
            weights = self.C.detach().cpu().numpy()
        elif x1 == self.layer_x[1]:  # Embedding to Hidden
            weights = self.W1.detach().cpu().numpy()
        else:  # Hidden to Output
            weights = self.W2.detach().cpu().numpy()
        
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
            
            # Draw character labels and probabilities for output layer
            if input_chars is not None and i < len(input_chars):
                char_text = self.font.render(input_chars[i], True, self.colors['text'])
                self.screen.blit(char_text, (x - 30, y - 8))
            elif output_chars is not None and i < len(output_chars):
                char_text = self.font.render(output_chars[i], True, self.colors['text'])
                self.screen.blit(char_text, (x + 15, y - 8))
                
                # Add probability distribution if we have logits
                if layer_name == 'output' and self.current_logits is not None:
                    probs = F.softmax(self.current_logits, dim=1)
                    prob = probs[0][i].item()
                    prob_text = f"{prob:.3f}"
                    prob_surface = self.font.render(prob_text, True, self.colors['text'])
                    self.screen.blit(prob_surface, (x + 45, y - 8))
                
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
            normalized = values / abs_max
        
        normalized = np.nan_to_num(normalized, nan=0.0)
        
        for pos, norm_value, orig_value in zip(positions, normalized, values):
            intensity = int(255 * abs(norm_value))
            
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
            
            # Draw the activation value and bias if this is the hidden layer
            if show_values:
                if layer_name == 'hidden':
                    # Get the corresponding bias value
                    bias_idx = positions.index(pos)
                    bias_value = self.b1[bias_idx].item()  # Changed indexing here
                    # Show both activation and bias
                    value_text = f"a:{orig_value:.2f}\nb:{bias_value:.2f}"
                else:
                    value_text = f"{orig_value:.2f}"
                
                # Split text into lines
                lines = value_text.split('\n')
                for i, line in enumerate(lines):
                    text = self.font.render(line, True, self.colors['text'])
                    text_pos = (pos[0] - text.get_width()/2, pos[1] - 45 + i*20)
                    self.screen.blit(text, text_pos)

    def _is_node_clicked(self, mouse_pos, node_pos):
        """Check if a node was clicked"""
        return ((mouse_pos[0] - node_pos[0])**2 + 
                (mouse_pos[1] - node_pos[1])**2 <= (self.node_radius * 2)**2)

    def process_custom_input(self, text, context_size, device):
        """Process custom input text into context"""
        # Ensure we have the stoi mapping
        if self.stoi is None:
            print("Error: Character to index mapping not available")
            return None
            
        # Convert input to indices, using 0 for unknown chars
        indices = []
        for char in text:
            if char in self.stoi:
                indices.append(self.stoi[char])
            else:
                print(f"Warning: Character '{char}' not in vocabulary, skipping")
                
        # If no valid characters, return
        if not indices:
            print("No valid characters found in input")
            return None
            
        # Pad to context_size if needed, or truncate
        if len(indices) < context_size:
            # Pad with zeros (usually representing space or start token)
            indices = [0] * (context_size - len(indices)) + indices
        elif len(indices) > context_size:
            # Take only the last context_size characters
            indices = indices[-context_size:]
            
        print(f"Processed custom input into context: {indices}")
        return indices

    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            
            # Handle text input mode
            if self.is_input_active:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if self.input_text.strip():  # Only process if there's actual input
                            self.custom_input = self.input_text
                            self.input_text = ""
                            self.is_input_active = False
                            self.has_new_custom_input = True
                            self.ready_for_next = True  # Allow progression to next step
                            print(f"Custom input accepted: '{self.custom_input}'")
                            return True
                        # If no input, stay in input mode
                        return True
                    elif event.key == pygame.K_BACKSPACE:
                        self.input_text = self.input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.input_text = ""
                        self.is_input_active = False
                        self.ready_for_next = True
                        return True
                    else:
                        self.input_text += event.unicode
                    # Redraw the input box
                    self.draw_network(self.C, self.W1, self.b1, self.W2, self.b2,
                                   current_context=self.current_context,
                                   current_emb=self.current_emb,
                                   current_h=self.current_h,
                                   current_logits=self.current_logits,
                                   itos=self.itos)
                    return True
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.button_rect.collidepoint(event.pos):
                    self.ready_for_next = True
                    return True
                elif self.custom_input_button.collidepoint(event.pos):
                    # Enter text input mode immediately
                    self.is_input_active = True
                    self.input_text = ""
                    # Redraw immediately to show input screen
                    self.draw_network(self.C, self.W1, self.b1, self.W2, self.b2,
                                   current_context=self.current_context,
                                   current_emb=self.current_emb,
                                   current_h=self.current_h,
                                   current_logits=self.current_logits,
                                   itos=self.itos)
                    return True
                elif self.embedding_button.collidepoint(event.pos):
                    self.show_embedding_view = not self.show_embedding_view
                    self.embedding_surface = None  # Force redraw
                    # Force immediate redraw
                    self.draw_network(self.C, self.W1, self.b1, self.W2, self.b2,
                                     current_context=self.current_context,
                                     current_emb=self.current_emb,
                                     current_h=self.current_h,
                                     current_logits=self.current_logits,
                                     itos=self.itos)
                    return True
                # Handle dimension selection clicks when embedding view is active
                if self.show_embedding_view:
                    for i in range(self.C.shape[1]):
                        btn = pygame.Rect(10 + i*30, self.height - 50, 25, 25)
                        if btn.collidepoint(event.pos):
                            # Toggle between dim1 and dim2
                            if i == self.dim1:
                                self.dim1 = self.dim2
                                self.dim2 = i
                            else:
                                self.dim1 = i
                            self.embedding_surface = None  # Force redraw
                            return True
                else:
                    # Check if any node was clicked
                    for layer_name, positions in zip(['input', 'embedding', 'hidden', 'output'], 
                                                  self.current_positions):
                        for i, pos in enumerate(positions):
                            if self._is_node_clicked(event.pos, pos):
                                # Toggle neuron state
                                if (layer_name, i) in self.disabled_neurons[layer_name]:
                                    self.disabled_neurons[layer_name].remove((layer_name, i))
                                else:
                                    self.disabled_neurons[layer_name].add((layer_name, i))
                                # Redraw the network with current state
                                self.draw_network(self.C, self.W1, self.b1, self.W2, self.b2, 
                                               current_context=self.current_context,
                                               current_emb=self.current_emb,
                                               current_h=self.current_h,
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
    # Load the model
    checkpoint = torch.load(filename)
    C = checkpoint['C']
    W1 = checkpoint['W1']
    b1 = checkpoint['b1']
    W2 = checkpoint['W2']
    b2 = checkpoint['b2']
    itos = checkpoint['itos']
    context_size = checkpoint['context_size']
    vocab_size = checkpoint['vocab_size']
    
    device = C.device
    embedding_dim = C.shape[1]
    hidden_size = W2.shape[0]
    
    # Print dimensions for debugging
    print(f"Model dimensions:")
    print(f"C shape: {C.shape} (vocab_size, embedding_dim)")
    print(f"W1 shape: {W1.shape}")
    print(f"W2 shape: {W2.shape}")
    print(f"Block size: {context_size}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Hidden size: {hidden_size}")
    
    # Reshape W1 to handle concatenated embeddings
    W1 = W1.expand(context_size * embedding_dim, hidden_size)
    
    print(f"W1 shape after reshape: {W1.shape}")
    
    # Print model dimensions for debugging
    print(f"Shapes before processing:")
    print(f"C: {C.shape}")
    print(f"W1: {W1.shape}")
    print(f"W2: {W2.shape}")
    print(f"Block size: {context_size}")
    
    device = C.device
    embedding_dim = C.shape[1]
    hidden_size = W2.shape[0]
    
    # Instead of reshaping W1, make sure it's the correct size
    expected_w1_shape = (context_size * embedding_dim, hidden_size)
    if W1.shape != expected_w1_shape:
        raise ValueError(f"W1 shape {W1.shape} doesn't match expected shape {expected_w1_shape}")
    
    # Add this after loading the weights
    W1 = W1.reshape(context_size * C.shape[1], -1)  # Reshape to [context_size * emb_dim, hidden_size]
    
    device = C.device
    similarity_dimensions = C.shape[1]
    
    print(f"Model loaded from {filename}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Using device: {device}")
    
    # Sample from the model
    if seed is None:
        g = torch.Generator(device=device).manual_seed(int(time.time()))
    else:
        g = torch.Generator(device=device).manual_seed(seed)
    
    visualizer = MLPVisualizer()
    
    print(f"Generating {num_samples} samples:")
    sample_count = 0
    
    while sample_count < num_samples:
        out = []
        context = [0] * context_size
        visualizer.current_sample = ""
        visualizer.has_new_custom_input = False
        
        running = True
        first_step = True
        
        while running:
            # Check for custom input on the first step of each sample
            if first_step and visualizer.has_new_custom_input and visualizer.custom_input:
                print(f"Processing custom input: '{visualizer.custom_input}'")
                custom_context = visualizer.process_custom_input(visualizer.custom_input, context_size, device)
                if custom_context:
                    context = custom_context
                    # Update the current sample to show the input
                    visualizer.current_sample = visualizer.custom_input
                    visualizer.custom_input = ""  # Clear after using
                    visualizer.has_new_custom_input = False
                    print(f"Starting with custom input context: {context}")
                    print(f"Current sample initialized to: '{visualizer.current_sample}'")
                    
                    # Force a redraw to show the initial state with custom input
                    context_chars = [itos[idx] for idx in context]
                    emb = C[torch.tensor(context, device=device)]
                    emb_flat = emb.view(-1).unsqueeze(0)
                    h = torch.tanh(emb_flat @ W1 + b1)
                    logits = h @ W2 + b2
                    
                    visualizer.draw_network(C, W1, b1, W2, b2,
                                         current_context=context_chars,
                                         current_emb=emb,
                                         current_h=h,
                                         current_logits=logits,
                                         itos=itos)
            
            # Convert context indices to characters for display
            context_chars = [itos[idx] for idx in context]
            
            # Get embeddings and reshape properly
            emb = C[torch.tensor(context, device=device)]  # Shape: [context_size, emb_dim]
            emb_flat = emb.view(-1).unsqueeze(0)  # Reshape to [1, context_size * emb_dim]

            # Forward pass
            h = torch.tanh(emb_flat @ W1 + b1)
            logits = h @ W2 + b2

            # Debug embeddings shape
            #print(f"Emb shape: {emb.shape}, Emb flat shape: {emb_flat.shape}")
            
            # Visualize the current state
            visualizer.draw_network(C, W1, b1, W2, b2,
                                 current_context=context_chars,
                                 current_emb=emb,  # Pass original embedding shape for visualization
                                 current_h=h,
                                 current_logits=logits,
                                 itos=itos)
            
            # Wait for user to click "Next Step"
            if not visualizer.wait_for_step():
                pygame.quit()
                return
            
            # If we entered text input mode, skip the rest of the loop
            if visualizer.is_input_active:
                first_step = True  # Reset first_step to try again with input
                continue
            
            # If this was the first step, mark it as complete
            first_step = False
                
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            if ix != 0:
                out.append(ix)
                visualizer.current_sample += itos[ix]  # Add new character to current sample
            if ix == 0:
                running = False
        
        print(f"{sample_count+1}. {''.join(itos[i] for i in out)}")
        sample_count += 1
    
    pygame.quit()
    # Return the loaded model components in case they're needed
    return {
        'C': C, 
        'W1': W1, 
        'b1': b1, 
        'W2': W2, 
        'b2': b2, 
        'itos': itos,
        'context_size': context_size,
        'vocab_size': vocab_size
    }

def load_model(filename, seed=None):
    # Load the model
    checkpoint = torch.load(filename)
    C = checkpoint['C']
    W1 = checkpoint['W1']
    b1 = checkpoint['b1']
    W2 = checkpoint['W2']
    b2 = checkpoint['b2']
    itos = checkpoint['itos']
    context_size = checkpoint['context_size']
    vocab_size = checkpoint['vocab_size']
    
    device = C.device
    embedding_dim = C.shape[1]
    hidden_size = W2.shape[0]
    
    # Print dimensions for debugging
    print(f"Model dimensions:")
    print(f"C shape: {C.shape} (vocab_size, embedding_dim)")
    print(f"W1 shape: {W1.shape}")
    print(f"W2 shape: {W2.shape}")
    print(f"Block size: {context_size}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Hidden size: {hidden_size}")
    
    # Reshape W1 to handle concatenated embeddings
    W1 = W1.expand(context_size * embedding_dim, hidden_size)
    
    print(f"W1 shape after reshape: {W1.shape}")
    
    # Print model dimensions for debugging
    print(f"Shapes before processing:")
    print(f"C: {C.shape}")
    print(f"W1: {W1.shape}")
    print(f"W2: {W2.shape}")
    print(f"Block size: {context_size}")
    
    device = C.device
    embedding_dim = C.shape[1]
    hidden_size = W2.shape[0]
    
    # Instead of reshaping W1, make sure it's the correct size
    expected_w1_shape = (context_size * embedding_dim, hidden_size)
    if W1.shape != expected_w1_shape:
        raise ValueError(f"W1 shape {W1.shape} doesn't match expected shape {expected_w1_shape}")
    
    # Add this after loading the weights
    W1 = W1.reshape(context_size * C.shape[1], -1)  # Reshape to [context_size * emb_dim, hidden_size]
    
    device = C.device
    similarity_dimensions = C.shape[1]
    
    print(f"Model loaded from {filename}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Using device: {device}")
    
    # Return the loaded model components in case they're needed
    return {
        'C': C, 
        'W1': W1, 
        'b1': b1, 
        'W2': W2, 
        'b2': b2, 
        'itos': itos,
        'context_size': context_size,
        'vocab_size': vocab_size
    }

def visualize_embedding_space(model_components, max_words=300, interactive=False, marker_size=120, text_size=10):
    """
    Visualize the embedding space of an MLP model with large, clear markers.
    Works with 1D, 2D, and higher-dimensional embeddings.
    
    Parameters:
    -----------
    model_components : dict
        Dictionary containing model components (as returned by load_model_and_sample)
    max_words : int
        Maximum number of words to display
    interactive : bool
        Whether to create an interactive plot (requires plotly)
    marker_size : int
        Size of the markers
    text_size : int
        Size of the text labels
        
    Returns:
    --------
    None (displays visualization)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter
    
    # Extract model components
    C = model_components['C'].cpu().detach().numpy()
    itos = model_components['itos']
    
    # Filter to max_words
    filtered_indices = list(range(min(len(itos), max_words)))
    
    # Skip special tokens (assuming token 0 is special)
    filtered_indices = [i for i in filtered_indices]
    
    # Get embeddings and words for filtered indices
    embeddings = C[filtered_indices]
    words = [itos[i] for i in filtered_indices]
    
    embedding_dim = embeddings.shape[1]
    print(f"Embedding dimension: {embedding_dim}")
    
    # Create colormap for better distinction
    colors = plt.cm.tab20(np.linspace(0, 1, len(embeddings)))
    
    if interactive:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            if embedding_dim == 1:
                # 1D embeddings
                y_values = np.zeros(len(embeddings))
                
                fig = go.Figure()
                
                # Add markers with characters inside
                fig.add_trace(go.Scatter(
                    x=embeddings.flatten(),
                    y=y_values,
                    mode='markers+text',
                    marker=dict(
                        size=marker_size,
                        color='lightblue',
                        line=dict(color='darkblue', width=2)
                    ),
                    text=words,
                    textposition='middle center',
                    textfont=dict(size=text_size, color='black'),
                    hoverinfo='text',
                    hovertext=[f'{word if word != '\n' else ""}: {emb[0]:.4f}' for word, emb in zip(words, embeddings)],
                    showlegend=False
                ))
                
                fig.update_layout(
                    title="1D Embedding Space Visualization",
                    xaxis_title="Embedding Value",
                    yaxis_title="",
                    yaxis=dict(
                        showticklabels=False,
                        zeroline=True,
                        range=[-1, 1]
                    ),
                    width=900,
                    height=500
                )
                
            elif embedding_dim == 2:
                # 2D embeddings
                fig = go.Figure()
                
                # Add markers with characters inside
                fig.add_trace(go.Scatter(
                    x=embeddings[:, 0],
                    y=embeddings[:, 1],
                    mode='markers+text',
                    marker=dict(
                        size=marker_size,
                        color='lightblue',
                        line=dict(color='darkblue', width=2)
                    ),
                    text=words,
                    textposition='middle center',
                    textfont=dict(size=text_size, color='black'),
                    hoverinfo='text',
                    hovertext=[f'{word}: ({emb[0]:.4f}, {emb[1]:.4f})' for word, emb in zip(words, embeddings)],
                    showlegend=False
                ))
                
                fig.update_layout(
                    title="2D Embedding Space Visualization",
                    xaxis_title="Embedding Dimension 1",
                    yaxis_title="Embedding Dimension 2",
                    width=900,
                    height=700
                )
                
            else:
                # For higher dimensions, use the first two dimensions
                print("Showing first two dimensions of higher-dimensional embeddings")
                fig = go.Figure()
                
                # Add markers with characters inside
                fig.add_trace(go.Scatter(
                    x=embeddings[:, 0],
                    y=embeddings[:, 1],
                    mode='markers+text',
                    marker=dict(
                        size=marker_size,
                        color='lightblue',
                        line=dict(color='darkblue', width=2)
                    ),
                    text=words,
                    textposition='middle center',
                    textfont=dict(size=text_size, color='black'),
                    hoverinfo='text',
                    hovertext=[f'{word}: ({emb[0]:.4f}, {emb[1]:.4f})' for word, emb in zip(words, embeddings)],
                    showlegend=False
                ))
                
                fig.update_layout(
                    title=f"{embedding_dim}D Embedding Space (First Two Dimensions)",
                    xaxis_title="Embedding Dimension 1",
                    yaxis_title="Embedding Dimension 2",
                    width=900,
                    height=700
                )
            
            fig.show()
            
        except ImportError:
            print("Plotly not installed. Falling back to matplotlib.")
            interactive = False
    
    if not interactive:
        plt.figure(figsize=(12, 10))
        
        if embedding_dim == 1:
            # 1D embeddings - plot on x-axis with y=0
            plt.scatter(embeddings.flatten(), np.zeros(len(embeddings)), 
                       s=marker_size*2, alpha=0.6, c='lightblue', edgecolors='darkblue', linewidths=2)
            plt.title("1D Embedding Space Visualization", fontsize=14)
            plt.xlabel("Embedding Value", fontsize=12)
            plt.ylabel("")
            plt.yticks([])  # Hide y-axis ticks
            plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            plt.ylim(-1, 1)
            
            # Add text labels inside the circles
            for i, word in enumerate(words):
                plt.text(embeddings[i, 0], 0, word, 
                        ha='center', va='center', 
                        fontsize=text_size, fontweight='bold')
            
        elif embedding_dim == 2:
            # 2D embeddings
            scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                                s=marker_size*2, alpha=0.6, c='lightblue', edgecolors='darkblue', linewidths=2)
            plt.title("2D Embedding Space Visualization", fontsize=14)
            plt.xlabel("Embedding Dimension 1", fontsize=12)
            plt.ylabel("Embedding Dimension 2", fontsize=12)
            
            # Add text labels inside the circles
            for i, word in enumerate(words):
                plt.text(embeddings[i, 0], embeddings[i, 1], word, 
                        ha='center', va='center', 
                        fontsize=text_size, fontweight='bold')
            
        else:
            # For higher dimensions, just use the first two dimensions
            print("Showing first two dimensions of higher-dimensional embeddings")
            scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                                s=marker_size*2, alpha=0.6, c='lightblue', edgecolors='darkblue', linewidths=2)
            plt.title(f"{embedding_dim}D Embedding Space (First Two Dimensions)", fontsize=14)
            plt.xlabel("Embedding Dimension 1", fontsize=12)
            plt.ylabel("Embedding Dimension 2", fontsize=12)
            
            # Add text labels inside the circles
            for i, word in enumerate(words):
                plt.text(embeddings[i, 0], embeddings[i, 1], word, 
                        ha='center', va='center', 
                        fontsize=text_size, fontweight='bold')
        
        plt.tight_layout()
        plt.grid(alpha=0.3)
        plt.show()

load_model_and_sample("MLP1.w", 20, 42)
    
model_components = load_model("MLP1.w", 42)
visualize_embedding_space(model_components)