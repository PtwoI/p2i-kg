"""OWL vocabulary, portable RDF data and SHACL constraints share one schema."""
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, DCTERMS
from rdflib.collection import Collection
from .schema import KINDS, RELATIONS

BASE = Namespace('https://github.com/PtwoI/p2i-kg#')
SH = Namespace('http://www.w3.org/ns/shacl#')
SKOS = Namespace('http://www.w3.org/2004/02/skos/core#')
PROV = Namespace('http://www.w3.org/ns/prov#')


def identifier(key):
    kind, name = key.split(':', 1)
    return URIRef(str(BASE) + kind + '/' + name)


def graph_base():
    graph = Graph()
    for prefix, ns in [('atlas', BASE), ('owl', OWL), ('rdfs', RDFS), ('sh', SH), ('skos', SKOS), ('prov', PROV)]:
        graph.bind(prefix, ns)
    return graph


def union(graph, kinds):
    if len(kinds) == 1:
        return BASE[kinds[0].capitalize()]
    node, head = BNode(), BNode()
    graph.add((node, RDF.type, OWL.Class))
    graph.add((node, OWL.unionOf, head))
    Collection(graph, head, [BASE[k.capitalize()] for k in kinds])
    return node


def vocabulary():
    graph = graph_base()
    graph.add((BASE['ontology'], RDF.type, OWL.Ontology))
    graph.add((BASE['ontology'], OWL.versionInfo, Literal('0.2')))
    for kind in (*KINDS, 'claim'):
        cls = BASE[kind.capitalize()]
        graph.add((cls, RDF.type, OWL.Class))
        graph.add((cls, RDFS.label, Literal(kind.capitalize())))
    # Models are individuals representing architectures, not owl:Class axioms.
    # extends is a sourced historical/structural claim and is not transitive.
    for relation, (domains, ranges) in RELATIONS.items():
        prop = BASE[relation]
        graph.add((prop, RDF.type, OWL.ObjectProperty))
        graph.add((prop, RDFS.domain, union(graph, domains)))
        graph.add((prop, RDFS.range, union(graph, ranges)))
    graph.add((BASE.subconcept_of, RDF.type, OWL.TransitiveProperty))
    graph.add((BASE.subconcept_of, RDFS.subPropertyOf, SKOS.broaderTransitive))
    graph.add((BASE.alternative_to, RDF.type, OWL.SymmetricProperty))
    return graph


def data_graph(catalog):
    graph = graph_base()
    for node in catalog.nodes:
        key = identifier(node.id)
        graph.add((key, RDF.type, BASE[node.kind.capitalize()]))
        graph.add((key, BASE.label, Literal(node.label)))
        graph.add((key, RDFS.label, Literal(node.label)))
        graph.add((key, BASE.summary, Literal(node.summary)))
        for alias in node.aliases:
            graph.add((key, SKOS.altLabel, Literal(alias)))
        for field in ('reference_year', 'family', 'runtime_id', 'runtime_note'):
            value = getattr(node, field)
            if value is not None:
                graph.add((key, BASE[field], Literal(value)))
        if node.url:
            graph.add((key, BASE.url, URIRef(str(node.url))))
        for author in node.authors:
            graph.add((key, BASE.author, Literal(author)))
    for edge in catalog.edges:
        source, target, claim = identifier(edge.source), identifier(edge.target), identifier(edge.claim_id)
        graph.add((source, BASE[edge.relation], target))
        graph.add((claim, RDF.type, BASE.Claim))
        for prop, value in [('source', source), ('relation', BASE[edge.relation]), ('target', target)]:
            graph.add((claim, BASE[prop], value))
        graph.add((claim, BASE.basis, Literal(edge.basis)))
        graph.add((claim, BASE.explanation, Literal(edge.explanation)))
        if edge.locator:
            graph.add((claim, BASE.locator, Literal(edge.locator)))
        for paper in edge.evidence:
            graph.add((claim, BASE.evidence, identifier(paper)))
            graph.add((claim, PROV.wasDerivedFrom, identifier(paper)))
    return graph


def shapes():
    graph = graph_base()
    def property_(shape, predicate, **constraints):
        prop = BNode()
        graph.add((shape, SH.property, prop))
        graph.add((prop, SH.path, predicate))
        for key, value in constraints.items():
            graph.add((prop, SH[key], value))
        return prop
    for kind in (*KINDS, 'claim'):
        shape = BASE[kind.capitalize() + 'Shape']
        graph.add((shape, RDF.type, SH.NodeShape))
        graph.add((shape, SH.targetClass, BASE[kind.capitalize()]))
        if kind != 'claim':
            property_(shape, BASE.label, minCount=Literal(1), maxCount=Literal(1), datatype=XSD.string, minLength=Literal(1))
            property_(shape, BASE.summary, minCount=Literal(1), maxCount=Literal(1), datatype=XSD.string, minLength=Literal(1))
        if kind == 'paper':
            property_(shape, BASE.url, minCount=Literal(1), nodeKind=SH.IRI)
            property_(shape, BASE.author, minCount=Literal(1), datatype=XSD.string)
            property_(shape, BASE.reference_year, minCount=Literal(1), datatype=XSD.integer)
        if kind == 'claim':
            for predicate in (BASE.source, BASE.relation, BASE.target):
                property_(shape, predicate, minCount=Literal(1), maxCount=Literal(1), nodeKind=SH.IRI)
            property_(shape, BASE.evidence, minCount=Literal(1), **{'class': BASE.Paper})
            property_(shape, BASE.explanation, minCount=Literal(1), datatype=XSD.string, minLength=Literal(1))
            prop = property_(shape, BASE.basis, minCount=Literal(1), maxCount=Literal(1))
            head = BNode()
            graph.add((prop, SH['in'], head))
            Collection(graph, head, [Literal(v) for v in ('documented', 'structural', 'pedagogical')])
    for relation, (domains, ranges) in RELATIONS.items():
        shape = BASE[relation + 'Shape']
        graph.add((shape, RDF.type, SH.NodeShape))
        graph.add((shape, SH.targetSubjectsOf, BASE[relation]))
        def allowed(shape_node, kinds):
            head = BNode()
            graph.add((shape_node, SH['or'], head))
            options = []
            for kind in kinds:
                option = BNode()
                graph.add((option, SH['class'], BASE[kind.capitalize()]))
                options.append(option)
            Collection(graph, head, options)
        allowed(shape, domains)
        prop = property_(shape, BASE[relation], nodeKind=SH.IRI)
        allowed(prop, ranges)
    return graph


def validate_rdf(catalog):
    from pyshacl import validate
    conforms, _, report = validate(data_graph(catalog), shacl_graph=shapes(), inference='none')
    return {'conforms': bool(conforms), 'report': report}
