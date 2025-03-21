import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import Counter
    
def visualize_embedding_space(model_components, method='tsne', interactive=False, min_freq=1, max_words=300):
    """
    Visualize the embedding space of an MLP model.
    Handles both 1D and higher-dimensional embedding spaces.
    
    Parameters:
    -----------
    model_components : dict
        Dictionary containing model components (as returned by load_model_and_sample)
    method : str
        Dimensionality reduction method, either 'tsne' or 'pca'
        (Only used if embedding dimension > 2)
    interactive : bool
        Whether to create an interactive plot (requires plotly)
    min_freq : int
        Minimum frequency of words to include in visualization
    max_words : int
        Maximum number of words to display
        
    Returns:
    --------
    None (displays visualization)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter
    
    # Extract model components
    C = model_components['C'].cpu().numpy()
    itos = model_components['itos']
    
    # Count word frequencies if the model has this information
    # If not, we'll just use all words up to max_words
    word_counts = Counter()
    for i in range(len(itos)):
        word_counts[i] = 1  # Default count
    
    # Filter words by frequency and limit to max_words
    filtered_indices = [i for i, count in word_counts.most_common() 
                        if count >= min_freq and i < len(itos)][:max_words]
    
    # Skip special tokens (assuming token 0 is special)
    filtered_indices = [i for i in filtered_indices if i != 0]
    
    # Get embeddings and words for filtered indices
    embeddings = C[filtered_indices]
    words = [itos[i] for i in filtered_indices]
    
    embedding_dim = embeddings.shape[1]
    print(f"Embedding dimension: {embedding_dim}")
    
    # Handle case based on embedding dimension
    if embedding_dim == 1:
        # For 1D embeddings, just plot directly
        print("Visualizing 1D embedding space...")
        reduced_embeddings = embeddings.flatten()
        
        # Create a second dimension for visualization (just zeros)
        reduced_embeddings = np.column_stack((reduced_embeddings, np.zeros_like(reduced_embeddings)))
    elif embedding_dim == 2:
        # For 2D embeddings, use directly
        print("Using original 2D embeddings...")
        reduced_embeddings = embeddings
    else:
        # Reduce dimensionality for higher dimensions
        if method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            print("Performing t-SNE dimensionality reduction...")
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        else:
            from sklearn.decomposition import PCA
            print("Performing PCA dimensionality reduction...")
            reducer = PCA(n_components=2)
        
        reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Create visualization
    if interactive:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            if embedding_dim == 1:
                # For 1D, use scatter plot with x-axis as embedding value and y-axis as 0
                fig = px.scatter(
                    x=reduced_embeddings[:, 0], 
                    y=reduced_embeddings[:, 1],
                    hover_name=words,
                    title="1D Embedding Space Visualization"
                )
                
                fig.update_layout(
                    xaxis_title="Embedding Value",
                    yaxis_title="",
                    yaxis=dict(
                        showticklabels=False,
                        zeroline=True,
                        range=[-0.5, 0.5]  # Constrain y-axis around zero
                    )
                )
            else:
                # For 2D or higher (reduced to 2D)
                fig = px.scatter(
                    x=reduced_embeddings[:, 0], 
                    y=reduced_embeddings[:, 1],
                    hover_name=words,
                    title=f"Embedding Space Visualization ({method.upper() if embedding_dim > 2 else '2D'})"
                )
                
                fig.update_layout(
                    xaxis_title=f"{'Embedding Dimension 1' if embedding_dim <= 2 else f'{method.upper()} Dimension 1'}",
                    yaxis_title=f"{'Embedding Dimension 2' if embedding_dim <= 2 else f'{method.upper()} Dimension 2'}"
                )
            
            # Add text labels
            fig.add_trace(go.Scatter(
                x=reduced_embeddings[:, 0],
                y=reduced_embeddings[:, 1],
                text=words,
                mode='markers+text',
                textposition='top center',
                textfont=dict(size=8),
                showlegend=False
            ))
            
            fig.update_layout(
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
            # For 1D, use scatter plot with x-axis as embedding value and y-axis as 0
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
            plt.title("1D Embedding Space Visualization")
            plt.xlabel("Embedding Value")
            plt.ylabel("")
            plt.yticks([])  # Hide y-axis ticks
            plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)  # Horizontal line at y=0
            plt.ylim(-0.5, 0.5)  # Constrain y-axis around zero
        else:
            # For 2D or higher (reduced to 2D)
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
            plt.title(f"Embedding Space Visualization ({method.upper() if embedding_dim > 2 else '2D'})")
            plt.xlabel(f"{'Embedding Dimension 1' if embedding_dim <= 2 else f'{method.upper()} Dimension 1'}")
            plt.ylabel(f"{'Embedding Dimension 2' if embedding_dim <= 2 else f'{method.upper()} Dimension 2'}")
        
        # Add labels for each point
        for i, word in enumerate(words):
            plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), 
                         fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.show()