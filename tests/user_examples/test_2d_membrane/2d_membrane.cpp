/* ---------------------------------------------------------------------------*
 *            SPHinXsys: 2D menmbrane example-one body version           *
 * ----------------------------------------------------------------------------*
 * This is the one of the basic test cases, also the first case for            *
 * understanding SPH method for solid simulation.                              *
 * In this case, the constraint of the beam is implemented with                *
 * internal constrained subregion.                                             *
 * ----------------------------------------------------------------------------*/
#include "particle_momentum_dissipation.h"
#include "particle_momentum_dissipation.hpp"
#include "porous_media_dynamics.h"
#include "porous_media_solid.h"
#include "porous_solid_particles.h"
#include "sphinxsys.h"
using namespace SPH;
//------------------------------------------------------------------------------
// global parameters for the case
//------------------------------------------------------------------------------
Real PL = 10.0;  // membrane length
Real PH = 0.125; // membrane thickenss
Real BC = PL * 0.15;

int num = 8;
// reference particle spacing
Real resolution_ref = PH / num;

/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vec2d(-PL, -PL),
                                 Vec2d(2.0 * PL, PL));

//----------------------------------------------------------------------
//	Material properties of the fluid.
//----------------------------------------------------------------------
Real rho_0 = 2.0; // // reference density non-dimensionlaize
Real poisson = 0.26316;
Real Youngs_modulus = 8.242e6;
Real physical_viscosity = 5000.0;

Real diffusivity_constant_ = 1.0e-4;
Real fulid_initial_density_ = 1.0;
Real water_pressure_constant_ = 3.0e6;
Real saturation = 0.4;

Real refer_density_energy = 0.5 * water_pressure_constant_  ;

//----------------------------------------------------------------------
//	Geometric shapes used in the system.
//----------------------------------------------------------------------
// a membrane base shape

std::vector<Vecd> beam_base_shape{
    Vecd(-resolution_ref * 3.0, -PH / 2.0), Vecd(-resolution_ref * 3.0, PH / 2.0), Vecd(0.0, PH / 2.0),
    Vecd(0.0, -PH / 2.0), Vecd(-resolution_ref * 3.0, -PH / 2.0)};

// a membrane shape
std::vector<Vecd> beam_shape{Vecd(0.0, -PH / 2.0), Vecd(0.0, PH / 2.0),
                             Vecd(PL, PH / 2.0), Vecd(PL, -PH / 2.0), Vecd(0.0, -PH / 2.0)};

// a membrane end shape
std::vector<Vecd> beam_end_shape{
    Vecd(PL, -PH / 2.0), Vecd(PL, PH / 2.0),
    Vecd(PL + 4.0 * resolution_ref, PH / 2.0), Vecd(PL + 4.0 * resolution_ref, -PH / 2.0),
    Vecd(PL, -PH / 2.0)};

// a membrane saturation shape
std::vector<Vecd> beam_saturation_shape{
    Vecd(PL / 2.0 - BC, 0.0), Vecd(PL / 2.0 - BC, PH / 2.0), Vecd(PL / 2.0 + BC, PH / 2.0),
    Vecd(PL / 2.0 + BC, 0.0), Vecd(PL / 2.0 - BC, 0.0)};

// Beam observer location
StdVec<Vecd> observation_location = {Vecd(PL / 4.0, 0.0)};

//----------------------------------------------------------------------
//	Define the beam body
//----------------------------------------------------------------------
class Beam : public MultiPolygonShape
{
  public:
    explicit Beam(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(beam_base_shape, ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(beam_shape, ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(beam_end_shape, ShapeBooleanOps::add);
    }
};

//----------------------------------------------------------------------
//	define the beam base which will be constrained.
//----------------------------------------------------------------------
MultiPolygon createBeamConstrainShape()
{
    MultiPolygon multi_polygon;
    multi_polygon.addAPolygon(beam_base_shape, ShapeBooleanOps::add);
    multi_polygon.addAPolygon(beam_shape, ShapeBooleanOps::sub);
    multi_polygon.addAPolygon(beam_end_shape, ShapeBooleanOps::add);
    return multi_polygon;
};

MultiPolygon createSaturationConstrainShape()
{
    MultiPolygon multi_polygon;
    multi_polygon.addAPolygon(beam_saturation_shape, ShapeBooleanOps::add);
    return multi_polygon;
};

//----------------------------------------------------------------------
//	application dependent initial condition
//----------------------------------------------------------------------
class SaturationInitialCondition : public  multi_species_continuum::PorousMediaSaturationDynamicsInitialCondition
{
  public:
    SaturationInitialCondition(BodyPartByParticle &body_part) :  multi_species_continuum::PorousMediaSaturationDynamicsInitialCondition(body_part){};
    virtual ~SaturationInitialCondition(){};

  protected:
    void update(size_t index_i, Real dt = 0.0)
    { 
        fluid_saturation_[index_i] = saturation;
        fluid_mass_[index_i] = saturation * fulid_initial_density_ * Vol_update_[index_i];
        total_mass_[index_i] = rho_n_[index_i] * Vol_update_[index_i] + fluid_mass_[index_i];
    };
};

//------------------------------------------------------------------------------
// the main program
//------------------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up the environment of a SPHSystem with global controls.
    //----------------------------------------------------------------------
    SPHSystem system(system_domain_bounds, resolution_ref);
#ifdef BOOST_AVAILABLE
    // handle command line arguments
    system.handleCommandlineOptions(ac, av);
#endif //----------------------------------------------------------------------
       //	Creating body, materials and particles.
       //----------------------------------------------------------------------
    SolidBody beam_body(system, makeShared<Beam>("2dMembrane"));
    beam_body.defineParticlesAndMaterial<multi_species_continuum::PorousMediaParticles, multi_species_continuum::PorousMediaSolid>(rho_0, Youngs_modulus, poisson,
                   diffusivity_constant_, fulid_initial_density_, water_pressure_constant_);
    beam_body.generateParticles<ParticleGeneratorLattice>();

    ObserverBody beam_observer(system, "MembraneObserver");
    beam_observer.defineAdaptationRatios(1.15, 2.0);
    beam_observer.generateParticles<ObserverParticleGenerator>(observation_location);
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //----------------------------------------------------------------------
    InnerRelation beam_body_inner(beam_body);
    ContactRelation beam_observer_contact(beam_observer, {&beam_body});
    //-----------------------------------------------------------------------------
    // this section define all numerical methods will be used in this case
    //-----------------------------------------------------------------------------

     
    // corrected strong configuration
    InteractionWithUpdate<CorrectedConfigurationInner> beam_corrected_configuration(beam_body_inner);
    // time step size calculation
    ReduceDynamics<solid_dynamics::AcousticTimeStepSize> computing_time_step_size(beam_body);
    ReduceDynamics< multi_species_continuum::GetSaturationTimeStepSize> saturation_time_step_size(beam_body);

    // stress relaxation for the beam
    Dynamics1Level<multi_species_continuum::PorousMediaStressRelaxationFirstHalf> stress_relaxation_first_half(beam_body_inner);
    Dynamics1Level<multi_species_continuum::PorousMediaStressRelaxationSecondHalf> stress_relaxation_second_half(beam_body_inner);
    Dynamics1Level<multi_species_continuum::SaturationRelaxationInPorousMedia> saturation_relaxation(beam_body_inner);

    // clamping a solid body part. This is softer than a direct constraint
    BodyRegionByParticle beam_base(beam_body, makeShared<MultiPolygonShape>(createBeamConstrainShape()));
    SimpleDynamics<multi_species_continuum::MomentumConstraint> clamp_constrain_beam_base(beam_base);

    BodyRegionByParticle beam_saturation(beam_body, makeShared<MultiPolygonShape>(createSaturationConstrainShape()));
    SimpleDynamics<SaturationInitialCondition>  constrain_beam_saturation(beam_saturation);
 
    // need to be done
    ReduceDynamics<TotalMechanicalEnergy> get_kinetic_energy(beam_body);

    /** Damping */
    DampingWithRandomChoice<InteractionSplit<multi_species_continuum::PorousMediaDampingPairwiseInner<Vec2d>>>
        beam_damping(0.5, beam_body_inner, "TotalMomentum", physical_viscosity);
    //-----------------------------------------------------------------------------
    // outputs
    //-----------------------------------------------------------------------------
    IOEnvironment io_environment(system);
    BodyStatesRecordingToVtp write_beam_states(io_environment, system.real_bodies_);
    // note there is a line observation
 
    ObservedQuantityRecording<Vecd>
        write_beam_tip_position("Position", io_environment, beam_observer_contact);

    //----------------------------------------------------------------------
    //	Setup computing and initial conditions.
    //----------------------------------------------------------------------
    system.initializeSystemCellLinkedLists();
    system.initializeSystemConfigurations();
    constrain_beam_saturation.exec();
    beam_corrected_configuration.exec();
    
    
    int ite = 0;
    int total_ite = 0; 
   
 
    GlobalStaticVariables::physical_time_ = 0.0;
     
    //----------------------------------------------------------------------
    //	Setup computing time-step controls.
    //----------------------------------------------------------------------
  

    Real End_Time = 100.0;
    Real setup_saturation_time_ = End_Time * 0.1;
   

    // time step size for output file
    Real D_Time = End_Time / 100.0;
    Real dt = 0.0; // default acoustic time step sizes

    // statistics for computing time
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    //-----------------------------------------------------------------------------
    // from here the time stepping begins
    //-----------------------------------------------------------------------------
    write_beam_states.writeToFile(0);
    write_beam_tip_position.writeToFile(0);
   
 
 
    // computation loop starts
    while (GlobalStaticVariables::physical_time_ < End_Time)
    {
        Real integration_time = 0.0;
        // integrate time (loop) until the next output time
        while (integration_time < D_Time)
        {
            Real Dt = saturation_time_step_size.exec();
            if (GlobalStaticVariables::physical_time_ < setup_saturation_time_)
            {
                constrain_beam_saturation.exec();
            }
            saturation_relaxation.exec(Dt);
            
            int stress_ite = 0;
            Real relaxation_time = 0.0;
            Real total_kinetic_energy = 1000.0;
	      

            while (relaxation_time < Dt)
            {
                if (total_kinetic_energy > (5e-9* refer_density_energy )) // this is because we change the total mehanical energy calculation
                {
                    stress_relaxation_first_half.exec(dt);
                    clamp_constrain_beam_base.exec();
                    beam_damping.exec(dt);
                    clamp_constrain_beam_base.exec();
                    stress_relaxation_second_half.exec(dt);

                    total_kinetic_energy = get_kinetic_energy.exec();
                    ite++;
                    stress_ite++;
                    dt = SMIN(computing_time_step_size.exec(), Dt);

                    if (ite % 1000 == 0)
                    {
                        std::cout << "N=" << ite << " Time: "
							<< GlobalStaticVariables::physical_time_ << "  Dt:" << Dt << "	dt: "
							<< dt << "  Dt/ dt:" << Dt / dt << "\n";               
                    }
                  
                }


                total_ite++;
                relaxation_time += dt;
                integration_time += dt;
                GlobalStaticVariables::physical_time_ += dt;
            }
                     
               std::cout << "One Diffusion finishes   "
                << "total_kinetic_energy =  " << total_kinetic_energy
                << "     stress_ite = " << stress_ite << std::endl;
        }

        TickCount t2 = TickCount::now();
        write_beam_states.writeToFile(ite);
        write_beam_tip_position.writeToFile(ite);
 
        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }
    TickCount t4 = TickCount::now();

    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds() << " seconds."
              << "  Iterations:  " << ite << std::endl;
    std::cout << "Total iterations computation:  " << GlobalStaticVariables::physical_time_ / dt
              << "  Total iterations:  " << total_ite << std::endl;

    return 0;
}
