# India state map data

`india_states_osm.geojson` is the locally bundled state and union-territory
boundary dataset used by `nirf/maps.py`. It contains 37 named geographic
features and stores each feature's display name in `properties.name`. The
shared map module explicitly maps the 33 canonical state/UT names in the NIRF
dataset to those GeoJSON names. Features without a matching NIRF state are
rendered as no-data areas.

The file is community-mapped OpenStreetMap boundary data, available under the
Open Database License (ODbL) 1.0. OpenStreetMap's copyright and license
information is at <https://www.openstreetmap.org/copyright>. This is **not**
an official Survey of India boundary source and must not be used as an
authoritative administrative or legal reference.

The GeoJSON is bundled so map rendering requires no runtime download.
