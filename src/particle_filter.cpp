/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  particles.resize( num_particles );
  weights.resize( num_particles );

  std::random_device rd{};
  std::mt19937 gen{rd()};

  std::normal_distribution<> dx{x, std[0]};
  std::normal_distribution<> dy{y, std[1]};
  std::normal_distribution<> dtheta{theta, std[2]};

  for( size_t i = 0; i < particles.size(); ++i ) {
      particles[i].id = i;
      particles[i].x = dx(gen);
      particles[i].y = dy(gen);
      particles[i].theta = normalize_angle( dtheta(gen) );
      weights[i] = 1.0;
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::random_device rd{};
  std::mt19937 gen{rd()};

  std::normal_distribution<> dx{0,      std_pos[0]};
  std::normal_distribution<> dy{0,      std_pos[1]};
  std::normal_distribution<> dtheta{0,  std_pos[2]};

  for( size_t i = 0; i < particles.size(); ++i ) {

    //this will "spread" the particles

    particles[i].x += dx( gen );
    particles[i].y += dy( gen );
    particles[i].theta += dtheta( gen );

    if( yaw_rate != 0 ) {

      //Adding some "noise" to the readings
      const double m = velocity/yaw_rate;

      //Let's operate through the particles. We will, first. create new readings with noise
      particles[i].x += m*( sin( particles[i].theta + yaw_rate*delta_t ) - sin( particles[i].theta ) );
      particles[i].y += m*( cos( particles[i].theta ) - cos( particles[i].theta + yaw_rate*delta_t ) );
      particles[i].theta = normalize_angle( particles[i].theta + yaw_rate*delta_t);

    }else {
      //Let's operate through the particles. We will, first. create new readings with noise
      particles[i].x += velocity*( cos( particles[i].theta ) )*delta_t;
      particles[i].y += velocity*( sin( particles[i].theta ) )*delta_t;
    }
  }
}

LandmarkObs ParticleFilter::dataAssociation(  const Map &map_landmarks,
                                              const LandmarkObs &predicted ) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  LandmarkObs n;
  n.id = -1;

  double min_dist = 1e6;
  for( size_t j = 0; j < map_landmarks.landmark_list.size(); ++j ) {
    double d = dist( predicted.x, predicted.y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f );

    if( d < min_dist ) {
      min_dist = d;
      n.x = map_landmarks.landmark_list[j].x_f;
      n.y = map_landmarks.landmark_list[j].y_f;
      n.id = map_landmarks.landmark_list[j].id_i;
    }
  }

  return n;

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  double weight_sum = 0.0;

  for( size_t  i = 0; i < particles.size(); ++ i ) {
    double xt = particles[i].x;
    double yt = particles[i].y;
    double theta = particles[i].theta;

    vector<LandmarkObs> mapObs;

    LandmarkObs tempMapObs;
    LandmarkObs associatedObs;

    particles[i].weight = 1.0;

    particles[i].associations.resize( observations.size() );
    particles[i].sense_x.resize( observations.size() );
    particles[i].sense_y.resize( observations.size() );

    for( size_t j = 0; j < observations.size(); ++j ) {
      tempMapObs = toMapFrame( xt, yt, theta, observations[j] );
      mapObs.push_back( tempMapObs );
      associatedObs = dataAssociation( map_landmarks, tempMapObs );

      particles[i].weight *= multiv_prob( std_landmark[0], std_landmark[1], tempMapObs.x, tempMapObs.y, associatedObs.x, associatedObs.y );

          
      //Set the association for the particle
      particles[i].associations[j] = associatedObs.id;
      particles[i].sense_x[j] = tempMapObs.x;
      particles[i].sense_y[j] = tempMapObs.y;
    }

    //Increment the normalizing summation
    weight_sum += particles[i].weight;
  }


  //Normalizing the weights
  for( size_t i = 0; i < particles.size(); ++i ) {
    particles[i].weight /= weight_sum;
    weights[i] = particles[i].weight;
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(weights.begin(), weights.end());

  std::vector<Particle> newParticles;
  newParticles.reserve( weights.size() );

  for( size_t n = 0; n < weights.size(); ++n ) {
    newParticles.push_back( particles[ d(gen) ] );
  }

  particles = newParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}


LandmarkObs ParticleFilter::toMapFrame( double xt, double yt, double theta, const LandmarkObs &obs ) {
  LandmarkObs transformedObs( obs );

  transformedObs.x = cos(theta)*obs.x - sin(theta)*obs.y + xt;
  transformedObs.y = sin(theta)*obs.x + cos(theta)*obs.y + yt;

  return transformedObs;
}