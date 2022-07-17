use rand::prelude::*;
use rand::distributions::StandardNormal;

#[derive(Clone)]
struct Particle{
    chromossome: Vec<f32>,
    wi: f32,
    wm: f32,
    wc: f32,
    pub fitness: i64,
    fitfunc: fn(&Vec<f32>) -> i64,
    mutation_rate: f32
}

impl Particle {
    pub fn new(chromossome: Vec<f32>, mutation_rate: &f32, wi:f32, wm:f32, wc:f32, fitfunc: fn(&Vec<f32>)->i64) -> Self {
        let mut ret = Self {
            chromossome,
            wi,
            wm,
            wc,
            fitness: 0,
            fitfunc,
            mutation_rate: *mutation_rate,
        };
        ret.estimate_fitness();
        return ret;
    }

    /// Introduces a mutation to the travel weights of a particle
    pub fn mutate(&mut self) {
        self.wi = self.wi * (1.0 + self.mutation_rate*SmallRng::from_entropy().sample(StandardNormal) as f32);
        self.wm = self.wm * (1.0 + self.mutation_rate*SmallRng::from_entropy().sample(StandardNormal) as f32);
        self.wc = self.wc * (1.0 + self.mutation_rate*SmallRng::from_entropy().sample(StandardNormal) as f32);
    }

    /// Calculates how fit the particle is according to it's genotype
    pub fn estimate_fitness(&mut self) {
        self.fitness = (self.fitfunc)(&self.chromossome);
    }
}

pub struct Swarm{
    particles: Vec<Particle>,
    ancestors: Vec<Particle>,
    best_ancestors: Vec<Particle>,
    global_best: Particle,
    gen: i32, // generation count
}

impl Swarm {
    pub fn new(
        size: &usize,
        chromossome_size: &usize,
        mutation_rate: &f32,
        wi: &f32,
        wm: &f32,
        wc: &f32,
        fitness_func: fn(&Vec<f32>) -> i64,
        ) -> Self {

        let mut particles = Vec::with_capacity(*size);
        let mut ancestors = Vec::with_capacity(*size);

        for i in 0..*size{
            let mut random_particle_chromossome: Vec<f32> = Vec::with_capacity(*chromossome_size);
            let mut random_ancestor_chromossome: Vec<f32> = Vec::with_capacity(*chromossome_size);

            for i in 0..*chromossome_size{
                random_particle_chromossome.push(rand::random());
                random_ancestor_chromossome.push(rand::random());
            }

            particles.push(
                Particle::new(random_particle_chromossome, mutation_rate, *wi, *wm,* wc, fitness_func)
            );
            ancestors.push(
                Particle::new(random_ancestor_chromossome, mutation_rate, *wi, *wm,* wc, fitness_func)
            );
        }

        let mut best_ancestors = ancestors.clone();

        let global_best = particles.iter().max_by_key(|particle| particle.fitness).unwrap().clone();

        Self { particles, ancestors, best_ancestors, global_best, gen: 1 }
    }

    /// Moves the particles around according to inertia, ancestors and global best
    pub fn travel(&mut self) {
        self.gen += 1;

        self.particles
            .iter_mut()
            .for_each(|particle: &mut Particle| 
                {
                    // TODO
                }
            )
    }
    
    /// Create a mutated child for each particle and adds it to the swarm (duplicates population)
    pub fn reproduce(&mut self) {
        // Clone current particle population i.e. each particle has a "son"
        let mut children: Vec<Particle> = self.particles.clone();
        
        // Mutate the children so it becomes different than the father
        // The child will have it's fitness evaluated only after having a chance to move
        children.iter_mut().for_each(
            |son: &mut Particle| 
                son.mutate()
            );

        // Insert children into particle swarm
        self.particles.append(&mut children);
        
        // The children will have the same ancestors and best ancestors
        let mut ancestors_copy: Vec<Particle> = self.ancestors.clone();
        let mut best_ancestors_copy: Vec<Particle> = self.best_ancestors.clone();

        self.ancestors.append(&mut ancestors_copy);
        self.best_ancestors.append(&mut best_ancestors_copy);
    }

    /// Kills the worst half of the particle population
    pub fn select(&mut self) {
        let size = self.particles.len() / 2;
        for i in 0..size {
            // Replace parent for child if it performs better
            if self.particles[i+size].fitness > self.particles[i].fitness {
                self.particles[i] = self.particles[i+size].clone();
                self.ancestors[i] = self.ancestors[i+size].clone();
                self.best_ancestors[i] = self.best_ancestors[i+size].clone();
            }
        }

        // Kill the worst half
        self.particles.truncate(size);
        self.ancestors.truncate(size);
        self.best_ancestors.truncate(size);
    }
}