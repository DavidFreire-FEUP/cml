use rand::prelude::*;
use rand::distributions::StandardNormal;

use crate::{mlp, consts::MUTATION_RATE};

use mlp::MLP;

struct Particle{
    chromossome: MLP,
    inertia: MLP,
    wi: f32,
    wm: f32,
    wc: f32,
    best_ancestor: MLP,
    fitness: f32
}

impl Particle {
    fn mutate(&mut self) {
        self.wi = self.wi * (1.0 + MUTATION_RATE*SmallRng::from_entropy().sample(StandardNormal) as f32);
        self.wm = self.wm * (1.0 + MUTATION_RATE*SmallRng::from_entropy().sample(StandardNormal) as f32);
        self.wc = self.wc * (1.0 + MUTATION_RATE*SmallRng::from_entropy().sample(StandardNormal) as f32);
    }

    fn estimate_fitness(&mut self) {
        // TODO
    }
}

struct Swarm{
    particles: Vec<Particle>,
    global_best: Particle,
    gen: i32, // generation count
}

impl Swarm {
    fn travel(&mut self) {
        self.particles
            .iter_mut()
            .for_each(|particle| 
                {
                    // TODO
                }
            )
    }
    
    fn reproduce(&mut self) {
        // TODO
    }

    fn select(&mut self) {
        // TODO
    }
}