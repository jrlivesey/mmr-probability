/**
 * @file mmrprob.c
 * @brief Determines from a set of posteriors how likely two planets in a
 *        system are in a mean motion resonance (MMR).
 * @author Joseph Livesey ([jrlivesey@wisc.edu](mailto:jrlivesey@wisc.edu))
 * @date 2024
*/

#include "rebound.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MSUN 1.988e+30 // Mass of Sun in kg
#define RSUN 6.957e+8 // Radius of Sun in m
#define MEARTH 5.970e+24 // Mass of Earth in kg
#define REARTH 6.371e+6 // Radius of Earth in m
#define MNEPTUNE 1.024e+26 // Mass of Neptune in kg
#define RNEPTUNE 2.460e+7 // Radius of Neptune in m
#define MJUPITER 1.898e+27 // Mass of Jupiter in kg
#define RJUPITER 6.995e+7 // Radius of Jupiter in m
                         //
#define NUMGRIDTIME 100 // Number of horizontal bins for resonance check
#define NUMGRIDRARG 100 // Number of vertical bins for resonance check
#define RARGDEPTH 4 // Number of permutations of resonant arguments to check
#define RANDOMSEED 42 // Seed for draws from posteriors

#define MSTAR 0
#define RHOSTAR 1
#define LOGROR 2
#define ESINW 3
#define ECOSW 4
#define INC 5
#define T0 6

#define MR_RELATION_TERRAN 0
#define MR_RELATION_NEPTUNIAN 1
#define MR_RELATION_JOVIAN 2
#define MR_RELATION_STELLAR 3

#define unif(a, b) rand() / ((double) RAND_MAX) * (b - a) + a


/* DATA STRUCTURES */

// typedef struct {
//     double *m_star;   // stellar mass
//     double *rho_star; // stellar density
//     double *log_ror;  // log(Rp/R*)
//     double *esinw;    // sqrt(e) * sin(omega)
//     double *ecosw;    // sqrt(e) * cos(omega)
//     double *inc;      // Inclinations
//     double *t0;       // Times of conjunction
// } POSTERIORS;

/**
 * @brief Contains posterior data for the system.
*/
typedef struct {
    int size;         /**< number of rows in the input file */
    int Nplanets;     /**< number of planets in the system */
    int Nobs;         /**< number of transits observed per planet */
    int ***indices;   /**< maps indices from arbitrary input order */
    char **columns;   /**< columns of the input file */
    double **links;   /**< posterior links */
} POSTERIORS;

/**
 * @brief Contains parameters, drawn from the posteriors, for a simulation.
*/
typedef struct {
    int isim;        /**< index of the simulation */
    int Nplanets;    /**< number of planets in the system */
    int Norbits;     /**< number of orbits to run simulations */
    int inner;       /**< The inner planet in the possible MMR */
    int outer;       /**< The outer planet in the possible MMR */
    double Pratio;   /**< Period ratio of the planets in possible MMR */
    double *m;       /**< masses */
    double *P;       /**< periods */
    double *e;       /**< eccentricities */
    double *w;       /**< arguments of pericenter */
    double *inc;     /**< inclinations (relative to fundamental plane) */
    double *M;       /**< mean anomalies */
} PARAMETERS;

/**
 * @brief Contains information to output to a file.
*/
typedef struct {
    int num;        /**< number of simulations to run */
    int *randlinks; /**< indices of the posterior links used to initialize sims */
    int **resonant; /**< is i-th simulation resonant for the j-th res argument? */
    int **coeffarr; /**< stores the coefficients of the resonant arguments used */
} OUTPUT;


/* ARRAY HELPER FUNCTIONS */

/**
 * Gets the length of an array (of doubles).
 * 
 * @param x The array in question.
*/
int lengthof_double(double *x) {
    int len = sizeof(*x) / sizeof(x[0]);
    return len;
}

/**
 * Gets the length of an array (of ints).
 * 
 * @param x The array in question.
*/
int lengthof_int(int *x) {
    int len = sizeof(*x) / sizeof(x[0]);
    return len;
}

/**
 * Gets the length (along the first axis) of an array (of ints).
 * 
 * @param x The array in question.
*/
int lengthof_int2d(int **x) {
    int len = sizeof(**x) / sizeof(x[0]);
    return len;
}

/**
 * Gets the length (in words, I guess?) of an array of strings.
 * 
 * @param x The array in question.
*/
int lengthof_string_words(char **x) {
    // TODO: Figure out how to do this; strlen() doesn't work for 2D
    int len = 0;
    while (x != NULL) {
        len++;
    }
    return len;
}

/**
 * Takes the mean of some 1D array.
 * 
 * @param x The array in question.
 * @return The mean of x.
*/
double mean(double *x) {
    double avg, sum = 0.0;
    int i, len = lengthof_double(x);
    for (i=0; i<len; i++) {
        sum += x[i];
    }
    avg = sum / len;
    return avg;
}

/**
 * Counts the number of `true` entries in a 1D boolean array.
 * 
 * @param x The array in question.
*/
int count_true(int *x) {
    int num_true = 0;
    int i, len = lengthof_int(x);
    for (i=0; i<len; i++) {
        if (x[i] == 1) {
            num_true++;
        }
    }
    return num_true;
}

/**
 * Counts the number of `true` entries in a 2D boolean array.
 * 
 * @param x The array in question.
*/
int count_true_2d(int **x) {
    int num_true = 0;
    int i, j;
    int xlen = lengthof_int2d(x);
    int ylen = lengthof_int(x[0]);
    
    for (i=0; i<xlen; i++) {
        for (j=0; j<ylen; j++) {
            if (x[i][j] == 1) {
                num_true++;
            }
        }
    }
    return num_true;
}

/**
 * Transposes a 2D array of ints.
 * 
 * @param src The input array.
 * @param dest The transpose of src.
*/
void transpose_int(int **src, int **dest) {
    int i, j, xlen, ylen;
    xlen = lengthof_int2d(src);
    ylen = lengthof_int(src[0]);
    
    for (i=0; i<xlen; i++) {
        for (j=0; j<ylen; j++) {
            dest[i][j] = src[j][i];
        }
    }
}


/* PHYSICS FUNCTIONS */

/**
 * Calculates a resonant argument.
 * 
 * @param o1 The orbit of the inner planet.
 * @param o2 The orbit of the outer planet.
 * @param coeff The list of four coefficients that specify the resonance.
 * @return The resonant argument.
*/
double resonant_argument(struct reb_orbit o1, struct reb_orbit o2,
                         int *coeff) {
    double arg;
    int i, dalembert = 0;
    for (i=0; i<4; i++) {
        dalembert += coeff[i];
    }
    if (dalembert != 1) exit(1);
    arg = coeff[0] * o1.l + coeff[1] * o2.l + coeff[2] * o1.pomega +
          coeff[3] * o2.pomega;
    return arg;
}

/**
 * Obtains the radius of a spherical object (star) from its mass and average
 * density.
 * 
 * @param mass The mass of the star.
 * @param density The density of the star.
 * @return The radius of the star, in corresponding units.
*/
double star_radius_from_density(double mass, double density) {
    double r3, radius;
    r3 = 3. * mass / (4. * M_PI * density);
    radius = pow(r3, 1./3.);
    return radius;
}

/**
 * Power law mass-radius relationship for a planet, from Chen & Kipping 2017:
 * R ~ M^p.
 * 
 * @param model The index of the model:
 *              "Terran worlds", p = 0.28,
 *              "Neptunian worlds", p = 0.59,
 *              "Jovian worlds", p = -0.04,
 *              "Stellar worlds", p = 0.88.
 * @param radius The radius of the planet in m.
 * @return The mass of the planet in kg.
*/
double planet_mass_from_radius(int model, double radius) {
    double mass, power;
    if (model == MR_RELATION_TERRAN) {
        power = 0.28;
        mass = pow(radius / REARTH, 1./power) * MEARTH;
    } else if (model == MR_RELATION_NEPTUNIAN) {
        power = 0.59;
        mass = pow(radius / RNEPTUNE, 1./power) * MNEPTUNE;
    } else if (model == MR_RELATION_JOVIAN) {
        power = -0.04;
        mass = pow(radius / RJUPITER, 1./power) * MJUPITER;
    } else if (model == MR_RELATION_STELLAR) {
        power = 0.88;
        mass = pow(radius / RSUN, 1./power) * MSUN;
    } else {
        exit(1);
    }
    return mass;
}


/* STATISTICS FUNCTIONS */

/**
 * Generates a random number from a normal distribution using the Box–Muller
 * transform (https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform).
 * 
 * @param m The mean of the distribution.
 * @param s The standard deviation.
 * @return A random number!
*/
double rv_normal(double m, double s) {
    double rv, r, theta, u1, u2;
    double z0;
    // double z1;
    u1 = unif(0., 1.);
    u2 = unif(0., 1.);
    r = sqrt(-2. * log(u1));
    theta = 2. * M_PI * u2;
    z0 = r * cos(theta);
    // z1 = r * sin(theta);
    // z0 and z1 are a pair of independent normal RVs, but we only need one
    // Convert from standard normal to the desired normal distribution
    rv = z0 * s + m;
    return rv;
}

/**
 * Finds the slope of a linear regression.
 * 
 * @param x The independent variable.
 * @param y The dependent variable.
 * @return The slope.
*/
double slope_best_fit(double *x, double *y) {
    int len = lengthof_double(x);
    double xx[len], xy[len];
    double meanx, meany, meanxx, meanxy, slope;
    for (int i=0; i<len; i++) {
        xx[i] = x[i] * x[i];
        xy[i] = x[i] * y[i];
    }
    meanx = mean(x);
    meany = mean(y);
    meanxx = mean(xx);
    meanxy = mean(xy);
    slope = (meanx * meany - meanxy) / (meanx * meanx - meanxx);
    return slope;
}

/**
 * Fits for the orbital period based on a set of measurements of the time of
 * conjunction.
 * 
 * @param t0 The times of conjunction.
 * @return The best-fit period.
*/
double period_best_fit(POSTERIORS *pos, double *t0) {
    double obs[pos->Nobs];
    double period_guess;

    period_guess = t0[1] - t0[0];
    for (int i=0; i<pos->Nobs; i++) {
        obs[i] = round((t0[i] - t0[0]) / period_guess);
    }
    return slope_best_fit(obs, t0);
}

/**
 * Helper function for reading in values from the input file.
 * 
 * @param src The input string.
 * @param dest The output array of numbers as doubles.
*/
void read_from_text(char *src, double *dest) {
    // for (int i=0; i<lengthof_string_words(src); i++) {
    //     sscanf(src[i+1], "%lf", &dest[i]);
    // }
    int i = 0;
    char *token = strtok(src, " ");
    while (token != NULL) {
        sscanf(token, "%lf", &dest[i]);
        token = strtok(NULL, " ");
        i++;
    }
}

/**
 * Assembles a data struct containing the observational posteriors for the
 * system from an input file.
 * 
 * @param infile The name of the input file giving posterior distributions.
 * @param pos The struct of posteriors to populate.
*/
void read_posteriors(char *infile, POSTERIORS *pos) {
    // double num_planets, num_obs;
    // char line[1024];
    // char *token;

    // FILE *fptr = fopen(infile, "r");
    // if (fptr == NULL) exit(1);
    // fgets(line, 1024, fptr);
    // token = strtok(line, " ");
    // // TODO: sizeof(token[0]) is not necessarily the size of each entry!!!
    // num_planets = (sizeof(token)/sizeof(token[0]) - 1) / 3;
    // pos->rho_star[3];
    // pos->log_ror[3*num_planets];
    // pos->esinw[3*num_planets];
    // pos->ecosw[3*num_planets];
    // pos->inc[3*num_planets];

    // while (fgets(line, 1024, fptr)) {
    //     token = strtok(line, " ");
    //     if (memcmp(token[0], "r", 1)) {
    //         read_from_text(token, pos->rho_star);
    //     } else if (memcmp(token[0], "l", 1)) {
    //         read_from_text(token, pos->log_ror);
    //     } else if (memcmp(token[0], "es", 2)) {
    //         read_from_text(token, pos->esinw);
    //     } else if (memcmp(token[0], "ec", 2)) {
    //         read_from_text(token, pos->ecosw);
    //     } else if (memcmp(token[0], "i", 1)) {
    //         read_from_text(token, pos->inc);
    //     } else if (memcmp(token[0], "t", 1)) {
    //         num_obs = ((sizeof(token)/sizeof(token[0]) - 1) / 3) / num_planets;
    //         pos->t0[3*num_planets*num_obs];
    //         read_from_text(token, pos->t0);
    //     } else {
    //         exit(1);
    //     }
    // }
    // fclose(fptr);

    FILE *fptr;
    char line[1024];
    int a=0, b=0, c=0, d=0, e=0, f=0;
    int i;

    fptr = fopen(infile, "r");
    if (fptr == NULL) exit(1);

    pos->Nplanets = 0;
    pos->Nobs = 0;
    // TODO: Does this work?
    for (i=0; i<lengthof_string_words(pos->columns); i++) {
        if (memcmp(pos->columns[i], "l", 1)) {
            pos->Nplanets++;
        }
    }
    for (i=0; i<lengthof_string_words(pos->columns); i++) {
        if (memcmp(pos->columns[i], "p", 1)) {
            pos->Nobs++;
        }
    }
    pos->Nobs /= pos->Nplanets;

    // pos->indices[7][pos->Nplanets][pos->Nobs] = {0};
    pos->indices = malloc(7 * pos->Nplanets * pos->Nobs * sizeof(int));
    for (i=0; i<lengthof_string_words(pos->columns); i++) {
        if (memcmp(pos->columns[i], "m", 1)) {
            pos->indices[MSTAR][0][0] = i;
        } else if (memcmp(pos->columns[i], "r", 1)) {
            pos->indices[RHOSTAR][0][0] = i;
        } else if (memcmp(pos->columns[i], "l", 1)) {
            pos->indices[LOGROR][a][0] = i;
            a++;
        } else if (memcmp(pos->columns[i], "es", 2)) {
            pos->indices[ESINW][b][0] = i;
            b++;
        } else if (memcmp(pos->columns[i], "ec", 2)) {
            pos->indices[ECOSW][c][0] = i;
            c++;
        } else if (memcmp(pos->columns[i], "i", 1)) {
            pos->indices[INC][d][0] = i;
            d++;
        } else if (memcmp(pos->columns[i], "p", 1)) {
            pos->indices[T0][e][f] = i;
            f++;
            if (f == pos->Nobs - 1) {
                e++;
                f = 0;
            }
        } else {
            fprintf(stderr, "ERROR: Unrecognized input parameter.\n");
            exit(1);
        }
    }

    i = 0;
    while (fgets(line, 1024, fptr)) {
        read_from_text(line, pos->links[i]);
        i++;
    }
    pos->size = i;

    fclose(fptr);
}

/**
 * Samples the given posterior distributions and converts everything into
 * parameters to initialize the rebound simulations.
 * 
 * @param pos The struct of posteriors.
 * @param par The struct of simulation parameters.
 * @param out The struct of output information.
*/
void sample_posteriors(POSTERIORS *pos, PARAMETERS *par, OUTPUT *out) {
    double uplim, r_star, radius;
    double t0[pos->Nobs];
    int randlink, used_before = 0;

    uplim = pos->size;
    randlink = unif(0., uplim);
    for (int i=0; i<par->isim; i++) {
        if (randlink == out->randlinks[i]) {
            used_before++;
            break;
        }
    }
    if (!used_before) {
        r_star = star_radius_from_density(
            pos->links[randlink][pos->indices[MSTAR][0][0]],
            pos->links[randlink][pos->indices[RHOSTAR][0][0]]
        );
        out->randlinks[par->isim] = randlink;
        par->Nplanets = pos->Nplanets;
        for (int i=0; i<pos->Nplanets; i++) {
            radius = r_star * pow(10.,
                    pos->links[randlink][pos->indices[LOGROR][i][0]]);
            par->m[i] = planet_mass_from_radius(MR_RELATION_NEPTUNIAN, radius);
            par->w[i] = atan2(
                pos->links[randlink][pos->indices[ESINW][i][0]],
                pos->links[randlink][pos->indices[ECOSW][i][0]]
            );
            par->e[i] = pos->links[randlink][pos->indices[ESINW][i][0]] /
                        sin(par->w[i]);
            par->inc[i] = pos->links[randlink][pos->indices[INC][i][0]];
            par->M[i] = unif(0., 2.*M_PI);
            for (int j=0; j<pos->Nobs; j++) {
                t0[j] = pos->links[randlink][pos->indices[T0][i][j]];
            }
            par->P[i] = period_best_fit(pos, t0);
            par->Pratio = par->P[par->outer-1] / par->P[par->inner-1];
        }
    } else {
        sample_posteriors(pos, par, out);
    }
}


/* SIMULATION FUNCTIONS */

/**
 * Runs a rebound simulation of the system, for a set of orbital parameters
 * drawn from the posteriors.
 * 
 * @param mstar The mass of the star, in simulation units.
 * @param par The struct of orbital parameters.
*/
void run_simulation(double mstar, PARAMETERS par) {
    int interval;
    struct reb_simulation *r = reb_simulation_create();

    reb_simulation_add_fmt(r, "m", mstar);
    for (int i=0; i<par.Nplanets; i++) {
        reb_simulation_add_fmt(r, "m P e omega M", par.m[i], par.P[i],
                               par.e[i], par.w[i], par.M[i]);
    }
    r->G = 1.0;
    r->integrator = REB_INTEGRATOR_WHFAST; // Upgrade to WHFast512 if possible
    r->exact_finish_time = 0;
    r->dt = 0.05 * par.P[0];
    interval = par.Norbits * par.P[par.inner-1] / 10 / NUMGRIDTIME;
    reb_simulation_save_to_file_interval(r, ".archive.bin", interval);
    reb_simulation_integrate(r, par.Norbits * par.P[par.inner-1]);
    reb_simulation_free(r);
}

/**
 * Finds the integer ratio p/q closest to the ratio of the planets' periods,
 * then finds the coefficients of the lowest-order arguments for that
 * resonance.
 * 
 * @param out The struct of output parameters.
 * @param max_den The maximum denominator we want in an integer ratio.
 * @param period_ratio The actual ratio of the planets' orbital periods.
*/
void identify_resonant_arguments(OUTPUT *out, int max_den,
                                 double period_ratio) {
    int floor_int, argnum, p, q, r, s;
    double floor_double, frac;

    // Use Farey sequence to get closest (reasonable) integer ratio
    floor_double = floor(period_ratio);
    floor_int = floor_double;
    frac = 1.0 / (period_ratio - floor_double);
    p = 0;
    q = 1;
    while (q/p < frac) {
        if (p > max_den) break;
        p++;
        q++;
    }
    p += q * floor_int;
    
    // Use d'Alembert's rule to find the other possible sets of coefficients
    argnum = 0;
    for (r=0; r<max_den; r++) {
        for (s=0; s<max_den; s++) {
            if (r + s == q - p) {
                out->coeffarr[argnum][0] = p;
                out->coeffarr[argnum][1] = -q;
                out->coeffarr[argnum][2] = -r;
                out->coeffarr[argnum][3] = -s;
                argnum++;
                break;
            }
        }
        if (argnum > RARGDEPTH) break;
    }
}

/**
 * Routine for counting points in bins on plot of resonant argument vs. time,
 * and from this procedure determining whether the system is in this resonance
 * according to our definition.
 * 
 * @param time Simulation times at which the resonant argument was evaluated.
 * @param arg The resonant arguments.
 * @return True if the system is in resonance, false if not.
*/
int is_in_resonance(double *time, double *arg) {
    int i, j, k, len = lengthof_double(time);
    double end_time = time[len-1];
    double time_bins[NUMGRIDTIME], arg_bins[NUMGRIDRARG];
    // int grid[NUMGRIDTIME][NUMGRIDRARG];
    int col[NUMGRIDTIME];
    int row[NUMGRIDRARG];

    for (i=0; i<NUMGRIDTIME; i++) {
        time_bins[i] = i * end_time / NUMGRIDTIME;
    }
    for (i=0; i<NUMGRIDRARG; i++) {
        arg_bins[i] = i * 2. * M_PI / NUMGRIDRARG;
    }

    for (i=0; i<NUMGRIDTIME; i++) {
        for (j=0; j<NUMGRIDRARG; j++) {
            for (k=0; k<len; k++) {
                if (time[k] > time_bins[i] && time[k] <= time_bins[i+1] &&
                    arg[k] > arg_bins[j] && arg[k] <= arg_bins[i+1]) {
                    // If there are any points in this grid box, change to true
                    // and move to the next grid box.
                    row[j] = 1;
                    break;
                } else {
                    row[j] = 0;
                }
            }
        }
        col[i] = count_true(row);
    }

    // If the argument librates more than X% of the time, then mark as resonant
    if (count_true(col) / NUMGRIDTIME >= 0.8) {
        return 1;
    } else {
        return 0;
    }
}

/**
 * Checks whether the system exhibits libration in a certain resonant argument
 * in a given rebound simulation.
 * 
 * @param par The struct of simulation parameters.
 * @param coeff The coefficients specifying the resonant argument.
*/
int check_resonance(PARAMETERS par, int *coeff) {
    struct reb_simulationarchive *sa;
    struct reb_simulation *r;
    struct reb_orbit o1, o2;
    int num_steps = 10 * NUMGRIDTIME;
    double time[num_steps], arg[num_steps];
    
    sa = reb_simulationarchive_create_from_file(".archive.bin");
    for (int i=0; i<num_steps; i++) {
        r = reb_simulation_create_from_simulationarchive(sa, i);
        if (r == NULL) exit(1);
        o1 = reb_orbit_from_particle(r->G, r->particles[0],
                                     r->particles[par.inner]);
        o2 = reb_orbit_from_particle(r->G, r->particles[0],
                                     r->particles[par.outer]);
        time[i] = r->t;
        arg[i]  = resonant_argument(o1, o2, coeff);
        reb_simulation_free(r);
    }
    return is_in_resonance(time, arg);
}

/**
 * Runs a suite of simulations with system parameters drawn from the provided
 * posteriors. Then, the likelihood that the system is in resonance is
 * evaluated.
 * 
 * @param num The number of simulations in the sample.
 * @param pos The struct of posteriors.
 * @param par The struct of simulation parameters.
 * @param out The struct of output information.
*/
void run_and_analyze_all(POSTERIORS *pos, PARAMETERS *par, OUTPUT *out) {
    out->randlinks = malloc(out->num * sizeof(int));
    out->resonant = malloc(out->num * RARGDEPTH * sizeof(int));
    for (int i=0; i<out->num; i++) {
        par->isim = i;
        sample_posteriors(pos, par, out);
        run_simulation(1., *par);
        identify_resonant_arguments(out, 5, par->Pratio);
        for (int j=0; j<RARGDEPTH; j++) {
            out->resonant[i][j] = check_resonance(*par, out->coeffarr[j]);
        }
    }
}

/**
 * Write the results of the whole experiment to a text file.
 * 
 * @param out The output struct.
 * @param num The number of simulations that were performed.
*/
void write_output(OUTPUT *out) {
    char *outfile = "mmr.out";
    double res_frac, res_for_arg;
    int **resonant_T = malloc(RARGDEPTH * out->num * sizeof(int));
    FILE *fptr = fopen(outfile, "w");

    transpose_int(out->resonant, resonant_T);
    res_frac = count_true_2d(out->resonant) / out->num;
    fprintf(fptr, "System is resonant in %f%% of trials.\n\n", res_frac * 100);
    for (int i=0; i<RARGDEPTH; i++) {
        res_for_arg = count_true(resonant_T[i]);
        fprintf(
            fptr, "p=%d, q=%d, r=%d, s=%d: %f%%", out->coeffarr[i][0],
            out->coeffarr[i][1], out->coeffarr[i][2], out->coeffarr[i][3],
            res_for_arg * 100
        );
    }
    fclose(fptr);
}

// void test_sim() {
//     struct reb_simulation *r = reb_simulation_create();
//     reb_simulation_add_fmt(r, "m", 1.);
//     reb_simulation_add_fmt(r, "m P h k", 1.0e-3, 1., 0.1, 0.);
//     reb_simulation_add_fmt(r, "m P h k", 1.0e-3, 10., 0.1, 0.);
//     reb_simulation_save_to_file_interval(r, "archive.bin", 10);
//     reb_simulation_integrate(r, 1e2);
// }


int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "ERROR: Expected 5 arguments, received %d.\n",
                argc - 1);
        fprintf(stderr, "USAGE: %s [name of input file] ", argv[0]);
        fprintf(stderr, "[inner resonant planet] [outer resonant planet] ");
        fprintf(stderr, "[number of simulations] [number of orbits per sim]");
        exit(1);
    }

    POSTERIORS pos;
    PARAMETERS par;
    OUTPUT out;
    srand(RANDOMSEED);

    sscanf(argv[2], "%d", &(&par)->inner);
    sscanf(argv[3], "%d", &(&par)->outer);
    sscanf(argv[4], "%d", &(&out)->num);
    sscanf(argv[5], "%d", &(&par)->Norbits);
    read_posteriors(argv[1], &pos);

    run_and_analyze_all(&pos, &par, &out);
    write_output(&out);

    // test_sim();
    return 0;
}
