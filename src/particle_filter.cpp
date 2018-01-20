/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

// Define a random engine
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    // Define the Number of Particles
    num_particles = 100;
    
    // Random Gaussian Noise Initial Definition
    normal_distribution<double> inoise_x(0, std[0]);
    normal_distribution<double> inoise_y(0, std[1]);
    normal_distribution<double> inoise_theta(0, std[3]);
    
    // Initialize All particles position and weights
    for (int i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = x;
        p.y = y;
        p.theta = theta;
        p.weight = 1.0;
        
        // Add the random noise
        p.x += inoise_x(gen);
        p.y += inoise_y(gen);
        p.theta += inoise_theta(gen);
        
        particles.push_back(p);
    }
    
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    // Define Random Gaussian Noise for Prediction
    normal_distribution<double> noise_x(0, std_pos[0]);
    normal_distribution<double> noise_y(0, std_pos[1]);
    normal_distribution<double> noise_theta(0, std_pos[2]);
    
    for (int i = 0; i < num_particles; i++) {
        
        // Predict Next State
        if (fabs(yaw_rate) > 0.0001) {
            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
            particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
            particles[i].theta += yaw_rate * delta_t;
        } else {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }
        
        // Add Gaussian Noise
        particles[i].x += noise_x(gen);
        particles[i].y += noise_y(gen);
        particles[i].theta += noise_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    for (int i = 0; i < observations.size(); i++) {
        
        // Initialize landmark distance and id
        int map_id = -1;
        double dist_err = numeric_limits<double>::max();
        
        for (int j = 0; j < predicted.size(); j++) {
            
            double error = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
            
            // Find the nearest landmark
            if (error < dist_err) {
                dist_err = error;
                map_id = predicted[j].id;
            }
        }
        
        // Set observation id to nearest landmark id
        observations[i].id = map_id;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    for (int i = 0; i < num_particles; i++) {
        
        // Particle Coordinates
        const double px = particles[i].x;
        const double py = particles[i].y;
        const double ptheta = particles[i].theta;
        
        // Define Map Landmark locations near the particle
        vector<LandmarkObs> predictions;
        
        // Iterate each map Landmark
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            // Get Landmark coordinates and id
            const double lmx = map_landmarks.landmark_list[j].x_f;
            const double lmy = map_landmarks.landmark_list[j].y_f;
            const int lmid = map_landmarks.landmark_list[j].id_i;
            
            const double dx = lmx - px;
            const double dy = lmy - py;
            
            // Check if landmark lies within sensor range
            if (sqrt(dx*dx+dy*dy) <= sensor_range) {
                predictions.push_back(LandmarkObs{lmid, lmx, lmy});
            }
        }
        
        // Populate a list of observations transformed to map coordinates
        vector<LandmarkObs> t_obs;
        for (int j = 0; j < observations.size(); j++) {
            double tx = cos(ptheta)*observations[j].x - sin(ptheta)*observations[j].y + px;
            double ty = sin(ptheta)*observations[j].x + cos(ptheta)*observations[j].y + py;
            t_obs.push_back(LandmarkObs{observations[j].id, tx, ty});
        }
        
        // Run the function dataAssociation
        dataAssociation(predictions, t_obs);
        
        // Update the particle weights
        particles[i].weight = 1.0;
        
        // Weight Calculation and Update
        for (int j = 0; j < t_obs.size(); j++) {
            
            // Get Observations
            double ox, oy, prx, pry;
            ox = t_obs[j].x;
            oy = t_obs[j].y;
            const int obs_id = t_obs[j].id;
            
            // Get prediction coordinates related to the observation
            for (int k = 0; k < predictions.size(); k++) {
                if (predictions[k].id == obs_id) {
                    prx = predictions[k].x;
                    pry = predictions[k].y;
                }
            }
            
            // Calculate the weights for the observation
            const double sx = std_landmark[0];
            const double sy = std_landmark[1];
            double wobs = (1/(2*M_PI*sx*sy)) * exp(-(pow(prx-ox,2)/(2*pow(sx, 2)) + (pow(pry-oy,2)/(2*pow(sy, 2)))));
            
            // Update particle weight
            particles[i].weight *= wobs;
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    vector<Particle> new_particles;
    
    // Populate Weights
    vector<double> weights;
    for (int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
    }
    
    // Generate Random Index
    uniform_int_distribution<int> uniintdist(0, num_particles-1);
    auto index = uniintdist(gen);
    
    // Calculate max weight
    double max_weight = *max_element(weights.begin(), weights.end());
    
    // Uniform Random distributions [0.0, max_weight)
    uniform_real_distribution<double> unirealdist(0.0, max_weight);
    
    double beta = 0.0;
    
    // Resample
    for (int i = 0; i < num_particles; i++) {
        beta += unirealdist(gen) * 2.0;
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
