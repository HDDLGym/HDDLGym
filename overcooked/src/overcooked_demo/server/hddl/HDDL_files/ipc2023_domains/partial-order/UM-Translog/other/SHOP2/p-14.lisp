(defproblem problem domain 
  (
    (Connects VendeeFrancheComteRoadRoute Vendee FrancheComte)
    (Available VendeeFrancheComteRoadRoute)
    (At_Vehicle LKW Vendee)
    (Available LKW)
    (PV_Compatible Tische LKW)
    (RV_Compatible VendeeFrancheComteRoadRoute LKW)
    (At_Package Tische Vendee)
    (type_Equipment_Position FrancheComte)
    (type_Equipment_Position LKW)
    (type_Equipment_Position Vendee)
    (type_Location FrancheComte)
    (type_Location Vendee)
    (type_Object LKW)
    (type_Object Tische)
    (type_Package Tische)
    (type_Package_Storage_Position FrancheComte)
    (type_Package_Storage_Position LKW)
    (type_Package_Storage_Position Vendee)
    (type_Parcels Tische)
    (type_Physical LKW)
    (type_Physical Tische)
    (type_Region FrancheComte)
    (type_Region Vendee)
    (type_Regular LKW)
    (type_Regular Tische)
    (type_Regular_Truck LKW)
    (type_Regular_Vehicle LKW)
    (type_Road_Route VendeeFrancheComteRoadRoute)
    (type_Route VendeeFrancheComteRoadRoute)
    (type_Thing FrancheComte)
    (type_Thing LKW)
    (type_Thing Tische)
    (type_Thing Vendee)
    (type_Thing VendeeFrancheComteRoadRoute)
    (type_Truck LKW)
    (type_Vehicle LKW)
    (type_Vehicle_Position FrancheComte)
    (type_Vehicle_Position Vendee)
    (type_sort_for_FrancheComte FrancheComte)
    (type_sort_for_LKW LKW)
    (type_sort_for_Tische Tische)
    (type_sort_for_Vendee Vendee)
    (type_sort_for_VendeeFrancheComteRoadRoute VendeeFrancheComteRoadRoute)
  )
  ((__top))
)