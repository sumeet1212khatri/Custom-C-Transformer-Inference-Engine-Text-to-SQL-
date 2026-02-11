/*
    FINAL UNIVERSAL C++ INFERENCE ENGINE (Windows/Linux Compatible)
    Author: Sumeet Khatri
    Description: Reads tokenized prompt from CLI, generates SQL.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// --- Configuration ---
struct Config {
    int n_layer;
    int n_head;
    int n_embd;
    int block_size;
    int vocab_size;
};

// --- Model Weights ---
struct TransformerWeights {
    float* token_embedding_table;
    float* pos_embedding_table;
    float* layers_base_ptr;
    float* ln_f_weight;
    float* ln_f_bias;
    float* wcls;
};

// --- Run State ---
struct RunState {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *logits;
};

// --- Tokenizer (Global) ---
char** vocab;

void load_tokenizer() {
    FILE *f = fopen("tokenizer.bin", "rb");
    if (!f) { printf("Error: tokenizer.bin not found!\n"); exit(1); }
    
    int vocab_size;
    if(fread(&vocab_size, sizeof(int), 1, f) != 1) { exit(1); }
    
    vocab = (char**)malloc(vocab_size * sizeof(char*));
    
    for (int i = 0; i < vocab_size; i++) {
        int len;
        if(fread(&len, sizeof(int), 1, f) != 1) { break; }
        vocab[i] = (char*)malloc(len + 1);
        if(fread(vocab[i], 1, len, f) != len) { break; }
        vocab[i][len] = '\0';
    }
    fclose(f);
}

// --- Math Kernels ---
void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n) -> xout (d)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void layernorm(float* o, float* x, float* weight, float* bias, int size) {
    float mean = 0.0f;
    for (int i = 0; i < size; i++) mean += x[i];
    mean /= size;
    float var = 0.0f;
    for (int i = 0; i < size; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= size;
    float std = sqrtf(var + 1e-5f);
    for(int i = 0; i < size; i++) {
        o[i] = ((x[i] - mean) / std) * weight[i] + bias[i];
    }
}

void gelu(float* x, int size) {
    for (int i = 0; i < size; i++) {
        float val = x[i];
        // Approximate GELU
        x[i] = 0.5f * val * (1.0f + tanhf(0.79788456f * (val + 0.044715f * val * val * val)));
    }
}

// --- Transformer Forward Pass ---
void transformer_forward(Config* p, TransformerWeights* w, RunState* s, int token, int pos) {
    int dim = p->n_embd;
    
    // 1. Embeddings
    float* content_row = w->token_embedding_table + token * dim;
    float* pos_row = w->pos_embedding_table + pos * dim;
    for(int i=0; i<dim; i++) s->x[i] = content_row[i] + pos_row[i];
    
    // 2. Layers
    float* l_ptr = w->layers_base_ptr;
    for (int l = 0; l < p->n_layer; l++) {
        // ATTENTION BLOCK
        float* ln1_w = l_ptr; l_ptr += dim;
        float* ln1_b = l_ptr; l_ptr += dim;
        layernorm(s->xb, s->x, ln1_w, ln1_b, dim);
        
        // Skip Attention Logic for this simplified demo (Pass-through)
        l_ptr += (3 * dim) * dim; // att w
        l_ptr += (3 * dim);       // att b
        l_ptr += dim * dim;       // proj w
        l_ptr += dim;             // proj b
        
        // MLP BLOCK
        float* ln2_w = l_ptr; l_ptr += dim;
        float* ln2_b = l_ptr; l_ptr += dim;
        layernorm(s->xb, s->x, ln2_w, ln2_b, dim);
        
        float* fc_w = l_ptr; l_ptr += (4 * dim) * dim;
        float* fc_b = l_ptr; l_ptr += (4 * dim);
        float* proj_w = l_ptr; l_ptr += dim * (4 * dim);
        float* proj_b = l_ptr; l_ptr += dim;
        
        matmul(s->hb, s->xb, fc_w, dim, 4*dim);
        for(int i=0; i<4*dim; i++) s->hb[i] += fc_b[i];
        gelu(s->hb, 4*dim);
        matmul(s->xb2, s->hb, proj_w, 4*dim, dim);
        for(int i=0; i<dim; i++) s->xb2[i] += proj_b[i];
        
        // Residual
        for(int i=0; i<dim; i++) s->x[i] += s->xb2[i];
    }
    
    // 3. Final Norm & Head
    w->ln_f_weight = l_ptr; l_ptr += dim;
    w->ln_f_bias = l_ptr; l_ptr += dim;
    layernorm(s->x, s->x, w->ln_f_weight, w->ln_f_bias, dim);
    
    w->wcls = l_ptr; 
    matmul(s->logits, s->x, w->wcls, dim, p->vocab_size);
}

// --- Main Driver ---
int main(int argc, char* argv[]) {
    // 1. Load Tokenizer
    load_tokenizer();
    
    // 2. Load Model (Using standard FILE I/O instead of mmap for Windows compatibility)
    FILE *f = fopen("model.bin", "rb");
    if (!f) { printf("Error: model.bin not found\n"); return 1; }
    
    Config config;
    if(fread(&config, sizeof(Config), 1, f) != 1) { return 1; }
    
    // Calculate file size and allocate memory
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET); // Reset to start
    
    // Skip config part to get to weights
    fseek(f, sizeof(Config), SEEK_SET);
    
    // Allocate memory for weights (Windows friendly)
    long weights_size = file_size - sizeof(Config);
    float* data = (float*)malloc(weights_size);
    if (!data) { printf("Error: Out of memory\n"); return 1; }
    
    // Read weights into memory
    if(fread(data, 1, weights_size, f) != weights_size) {
        printf("Error reading weights\n");
        return 1;
    }
    fclose(f);
    
    // Setup Pointers
    TransformerWeights w;
    float* ptr = data;
    w.token_embedding_table = ptr; ptr += config.vocab_size * config.n_embd;
    w.pos_embedding_table = ptr; ptr += config.block_size * config.n_embd;
    w.layers_base_ptr = ptr;
    
    // Setup Run State
    RunState s;
    s.x = (float*)malloc(config.n_embd * sizeof(float));
    s.xb = (float*)malloc(config.n_embd * sizeof(float));
    s.xb2 = (float*)malloc(config.n_embd * sizeof(float));
    s.hb = (float*)malloc(4 * config.n_embd * sizeof(float));
    s.logits = (float*)malloc(config.vocab_size * sizeof(float));
    
    // --- PROMPT PROCESSING ---
    int prompt_tokens[256];
    int num_prompt = 0;
    
    if (argc > 1) {
        char* token = strtok(argv[1], ",");
        while(token != NULL && num_prompt < 256) {
            prompt_tokens[num_prompt++] = atoi(token);
            token = strtok(NULL, ",");
        }
    } else {
        prompt_tokens[0] = 50256; // Default
        num_prompt = 1;
    }
    
    // 1. Prefill (Consume Prompt)
    int token = prompt_tokens[0];
    int pos = 0;
    for(int i=0; i < num_prompt - 1; i++) {
        transformer_forward(&config, &w, &s, prompt_tokens[i], pos++);
    }
    token = prompt_tokens[num_prompt - 1]; // Start generation
    
    // 2. Generate
    for(int i=0; i<30; i++) {
        transformer_forward(&config, &w, &s, token, pos++);
        
        float max_val = -1e9;
        int next_token = 0;
        for(int j=0; j<config.vocab_size; j++) {
            if (s.logits[j] > max_val) { max_val = s.logits[j]; next_token = j; }
        }
        
        printf("%s", vocab[next_token]); // Output word
        fflush(stdout); 
        
        token = next_token;
        if (token == 50256) break; // Stop at EOS
    }
    
    return 0;
}