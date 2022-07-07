#include <stdio.h>

typedef struct
{
    float * value;
    int len;
} ColumnVector;

// n x m matrix (n rows by m columns)
typedef struct
{
    float ** value;
    int n;
    int m;
} Matrix;

typedef struct {
    MLP * chromossome;
    MLP inertia;
    MLP best_ancestor;
    int fitness;
} Particle;

typedef struct {
    int size;
    Particle *particles;
} Swarm;

Swarm * create_swarm(int size, MLP* mlp){
    Swarm * s = (Swarm*)malloc(sizeof(Swarm));
    s->size = size;
    s->particles = (Particle*)malloc(sizeof(Particle)*size);
    //TODO initialize particles
    return s;
}

void delete_swarm(Swarm* s){
    for (size_t i = 0; i < s->size; i++)
    {
        if(s->particles[i].chromossome != NULL) 
            delete_mlp(s->particles[i].chromossome);  
    }
    if(s->particles != NULL) 
        free(s->particles);
    if(s!=NULL) 
        free(s);

    return;
}

void move_swarm(Swarm *s){
    for(size_t i = 0; i<s->size; i++){
        //TODO
    }
}

typedef struct {
    float *values;
    Matrix weights;
    Matrix bias;
    void *activation; // activation function pointer
} NeuronLayer;

typedef struct{
    int hidden_ammount;
    NeuronLayer* layers;
} MLP;

MLP* create_mlp(int input_size, int output_size, int hidden_ammount, int hidden_size, void* activation){
    MLP* m = (MLP*)malloc(sizeof(MLP));

    // alloc size for hidden layers + output layer (input doesn't have weight, bias nor activation)
    m->layers = (NeuronLayer*)malloc(sizeof(NeuronLayer)*(2+hidden_ammount));

    for(size_t i=0; i<hidden_ammount+2; i++){
        // int wb_size[2] = {hidden_size};
        // int v_size = hidden_size;
        // if(i==0){
        //     wb_size = {0,0};
        //     v_size = input_size;
        // }
        // else if (i==1)
        //     wb_size=
        // }
        
        // Matrix* weights = (Matrix*)malloc(sizeof(float)*wb_size);
        // Matrix* bias = (Matrix*)malloc(sizeof(float)*wb_size);
        // ColumnVector* values = (float**)malloc(sizeof(float)*v_size);
        // for (size_t j = 0; j < wb_size; j++) {
        //     weights[j] = rand()%10;
        //     bias[j] = 0;
        // }
        // for (size_t j = 0; j < v_size; j++) values[j] = 0;

        // m->layers[i].weights = weights;
        // m->layers[i].bias = bias;
        // m->layers[i].values = values;
        m->layers[i].activation = activation;
    }
    
    m->hidden_ammount = hidden_ammount;
    return m;
}

void delete_mlp(MLP* m){
    //TODO
    return;
}

float* mlp_inout(MLP* m, float* input){
    //TODO
    return;
}

float relu(float input){
    if(input>1) return 1;
    else if(input<0) return 0;
    return input;
}

int main(){
    printf("Hello world\n");
    return 0;
