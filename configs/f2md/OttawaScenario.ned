import org.car2x.veins.nodes.Scenario;

network OttawaScenario extends Scenario
{
    string traceJSONFile = default("JSONlog-");
    string traceGroundTruthJSONFile = default("GroundTruthJSONlog-");
}
