# City Tree Inventories

The 'tree_inventory/code' module provides scripts for processing city tree inventory datasets, and identifying tree genera and tree species from GSV images.

## Files

- **AutoArborist_Species_Level_Images**: Verify matching between tree genera and tree species based on city inventories.
- **inventory_autoarborist_merger**: takes the AutoArborist dataset and attempts to match each longitude and latitude pair with those from the city data to rectify species-level images.
- **inventory_name_appender**: Creates a cleaned version of the genus and species for each organism detected
