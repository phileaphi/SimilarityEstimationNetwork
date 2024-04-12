
  (function() {
    var fn = function() {
      Bokeh.safely(function() {
        (function(root) {
          function embed_document(root) {
            
          var docs_json = '{"a72d2477-b63b-47a8-8882-81791a895d41":{"roots":{"references":[{"attributes":{},"id":"2854","type":"StringEditor"},{"attributes":{},"id":"2859","type":"StringFormatter"},{"attributes":{},"id":"2855","type":"StringFormatter"},{"attributes":{},"id":"2857","type":"StringFormatter"},{"attributes":{"columns":[{"id":"2847","type":"TableColumn"},{"id":"2848","type":"TableColumn"},{"id":"2849","type":"TableColumn"}],"height":140,"index_position":null,"source":{"id":"2846","type":"ColumnDataSource"},"view":{"id":"2851","type":"CDSView"}},"id":"2850","type":"DataTable"},{"attributes":{},"id":"2853","type":"UnionRenderers"},{"attributes":{"callback":null,"data":{"LPI":["71.60 +/- 42723.20","28.40 +/- 46050.60","00.00 +/- 38.56","00.00 +/- 38.56"],"Parameters":["model","optim_args.weight_decay","warmstart","transforms"],"fANOVA":["47.39 +/- 40.15","22.16 +/- 37.82","14.16 +/- 30.30","06.21 +/- 24.05"]},"selected":{"id":"2852","type":"Selection"},"selection_policy":{"id":"2853","type":"UnionRenderers"}},"id":"2846","type":"ColumnDataSource"},{"attributes":{"default_sort":"descending","editor":{"id":"2858","type":"StringEditor"},"field":"LPI","formatter":{"id":"2859","type":"StringFormatter"},"title":"LPI","width":100},"id":"2849","type":"TableColumn"},{"attributes":{"source":{"id":"2846","type":"ColumnDataSource"}},"id":"2851","type":"CDSView"},{"attributes":{},"id":"2858","type":"StringEditor"},{"attributes":{"default_sort":"descending","editor":{"id":"2854","type":"StringEditor"},"field":"Parameters","formatter":{"id":"2855","type":"StringFormatter"},"sortable":false,"title":"Parameters","width":150},"id":"2847","type":"TableColumn"},{"attributes":{"default_sort":"descending","editor":{"id":"2856","type":"StringEditor"},"field":"fANOVA","formatter":{"id":"2857","type":"StringFormatter"},"title":"fANOVA","width":100},"id":"2848","type":"TableColumn"},{"attributes":{},"id":"2852","type":"Selection"},{"attributes":{},"id":"2856","type":"StringEditor"}],"root_ids":["2850"]},"title":"Bokeh Application","version":"1.1.0"}}';
          var render_items = [{"docid":"a72d2477-b63b-47a8-8882-81791a895d41","roots":{"2850":"74ef01cf-23ee-4a57-86ed-7060324f25b8"}}];
          root.Bokeh.embed.embed_items(docs_json, render_items);
        
          }
          if (root.Bokeh !== undefined) {
            embed_document(root);
          } else {
            var attempts = 0;
            var timer = setInterval(function(root) {
              if (root.Bokeh !== undefined) {
                embed_document(root);
                clearInterval(timer);
              }
              attempts++;
              if (attempts > 100) {
                console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                clearInterval(timer);
              }
            }, 10, root)
          }
        })(window);
      });
    };
    if (document.readyState != "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  })();
