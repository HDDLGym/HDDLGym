#! /usr/bin/env python3

#This is a random problem generator (inspired by the problem-generator used for IPC2020).

import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("outputfile", type=str)
    parser.add_argument("--num_landingSpots", default=1, type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    problem = """
(define (problem pilotfit)
    (:domain lowfuel)
    (:objects
		; ----- cabin accessories
		fireExtinguisher - FireExtinguisher
		; ----- aircraft parts
		magnetos - Magnetos 
		engine - Engine
		fuelGauge - FuelGauge
		master - Master
		mixtureControl - MixtureControl
		throttle - Throttle
		heater - CabinHeater
		cabin - Cabin
		airvent - Airvent
		flaps - Flaps
		landingGear - LandingGear
		safetyBelt - SafetyBelt
		stick - Stick
		trim - Trim
		aileron - Aileron
		rudder - Rudder
		electricalSystem - ElectricalSystem		
		nose - Nose
		; ----- message to the ATC
		mayday - MaydayMessage
		fireOnStartMessage - FireOnStartMessage 
		electricalFireOnGroundMessage - ElectricalFireOnGroundMessage 
		electricalFireOnFlightMessage - ElectricalFireOnFlightMessage 
		cabinFireOnFlightMessage - CabinFireOnFlightMessage
		; ----- control parameters			 
		heading - Heading
        pitchRate - PitchRate
        correctCruisingAirSpeed - Airspeed
		speedZeroKmh - AirspeedHalt
		airspeed120kmh - AirspeedForLanding
		airspeed100kmh - AirspeedOnFinal
		airspeed125kmh - AirSpeedFlyOver
		groundSpeed90kmh - GroundSpeedTakeoff
		speed115kmh - AirspeedTakeOffClimb
		minAltitude150ft - AltitudeTakeOffClimb
        propellerRatio - PropellerRatio
        cruisingAltitude - Altitude
		altitude150ft - AltitudeFlyOver
		power3000rpm - PowerBeforeLanding
		altitude30ft - LandingAltitude1
		altitude2ft - LandingAltitude2
		thrustLevel - Thrust
		throttleRatioTakeOffClimb - ThrottleRatioTakeOffClimb
		; ----- Parts positions
		flapsOnPositionOne - FlapsPositionOne
		flapsLandingPosition - FlapsLandingPosition
		flapsPositionUp - FlapsPositionUp
		stickBackwards - StickLandingPosition
		aileronNeutralPosition - AileronNeutralPosition
		rudderOppositePosition - RudderOppositePosition
		trimNeutral - TrimAfterLandingPosition
		noseDown - NoseDown
		noseLiftedSlightly - NoseLiftedSlightly
		; ----- External conditions & locations
		windSpeedCondition - WindSpeedCondition
		windDirectionCondition - WindDirectionCondition
		headingCondition - HeadingCondition
		airspeedCondition - AirspeedCondition
        aerodrome1 - LandingSpot
		runwayAerodrome1 - Runway
		aerodrome1SurfaceCondition - LandingSpotSurfaceCondition
		aerodrome1SlopeCondition - LandingSpotSlopeCondition
		aerodrome1ObstructionCondition - LandingSpotObstructionCondition
        {}
    )	
    (:htn
        :tasks(and (precautionary_land))
        :ordering()
        :constraints()
    )
    (:init
        ; turned on during in-flight
		(p_isOn engine)
		(p_isOn mixtureControl)
		(p_pilotInAirplane)
		(p_isOn fuelGauge)
		(p_isOn magnetos)
		(p_isOn master)
		(p_isOpen throttle)
		(p_isOn heater)
		(not (p_isOpen airvent))
		(p_extinguisherExist fireExtinguisher)	
		; flight phases	
		(p_landingSpotClear aerodrome1)
		(not (p_reachable aerodrome1))
		(p_acceptableLandingCondition aerodrome1SurfaceCondition aerodrome1)
		(p_acceptableLandingCondition aerodrome1SlopeCondition aerodrome1)
		(p_acceptableLandingCondition aerodrome1ObstructionCondition aerodrome1)
        {}
    )
    (:goal
		(and
			(p_airplaneLanded)
		)
	)
)
    """
    objects = []
    init = []

    for i in range(1, args.num_landingSpots + 1):
        objects.extend(
            [
                f"aerodrome{i} - LandingSpot",
                f"runwayAerodrome{i} - Runway",
                f"aerodrome{i}SurfaceCondition - LandingSpotSurfaceCondition",
                f"aerodrome{i}SlopeCondition - LandingSpotSlopeCondition",
                f"aerodrome{i}ObstructionCondition - LandingSpotObstructionCondition",
            ]
        )
        if random.randint(0, 1)==0:
            init.extend(
                [
                    f"(p_landingSpotClear aerodrome{i})",
                ]
            )
        else:
            init.extend(
                [
                    f"(not (p_landingSpotClear aerodrome{i}))",
                ]
            )
        if random.randint(0, 1)==0:
            init.extend(
                [
                    f"(p_reachable aerodrome{i})",
                ]
            )
        else:
            init.extend(
                [
                    f"(not (p_reachable aerodrome{i}))",
                ]
            )
        if random.randint(0, 1)==0:
            init.extend(
                [
                    f"(p_acceptableLandingCondition aerodrome{i}SurfaceCondition aerodrome{i})",
                ]
            )
        else:
            init.extend(
                [
                    f"(not (p_acceptableLandingCondition aerodrome{i}SurfaceCondition aerodrome{i}))",
                ]
            )
        if random.randint(0, 1)==0:
            init.extend(
                [
                    f"(p_acceptableLandingCondition aerodrome{i}SlopeCondition aerodrome{i})",
                ]
            )
        else:
            init.extend(
                [
                    f"(not (p_acceptableLandingCondition aerodrome{i}SlopeCondition aerodrome{i}))",
                ]
            )
        if random.randint(0, 1)==0:
            init.extend(
                [
                    f"(p_acceptableLandingCondition aerodrome{i}ObstructionCondition aerodrome{i})",
                ]
            )
        else:
            init.extend(
                [
                    f"(not (p_acceptableLandingCondition aerodrome{i}ObstructionCondition aerodrome{i}))",
                ]
            )

    with open(args.outputfile, "w+") as f:
        f.write(
            problem.format(
                "\n".join(objects), "\n".join(init)
            )
        )


if __name__ == "__main__":
    main()
