import os
import json
import datetime
from typing import List, Dict, Any
import markdown

class HtmlReporter:
    """Generates an HTML report for agent regression tests."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_report(self, results: List[Dict[str, Any]], agent_class_name: str) -> str:
        """
        Generates a detailed HTML report from test results.

        Creates a professional HTML report with:
        - Summary statistics: total count, pass/fail counts, pass rate percentage
        - Results table: scenario details, pass/fail status, scores, metrics badges
        - Expandable details: input/output, metric explanations, token usage, errors
        - Markdown rendering: agent outputs rendered as HTML for better readability
        - Responsive design: works on desktop and mobile
        - Timestamp: report generation time included in filename and footer

        Result dict format expected:
        {
            'scenario': str,
            'agent_name': str,
            'model_id': str,
            'passed': bool,
            'score': float (0-10),
            'input': str,
            'actual_output': str,
            'expected_output': str,
            'explanation': str,
            'metrics': {metric_name: {score: float, explanation: str}},
            'token_usage': dict (optional),
            'error': str (optional)
        }

        Args:
            results: List of test result dictionaries (one per scenario evaluation).
            agent_class_name: Agent class name for report title and filename.

        Returns:
            Path to the generated HTML file (report_{agent_class_name}_{timestamp}.html).

        Note:
            Creates output directory if it doesn't exist.
            Filename includes timestamp to avoid overwrites across multiple runs.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"report_{agent_class_name}_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)

        passed_count = sum(1 for r in results if r.get('passed', False))
        failed_count = len(results) - passed_count
        total_count = len(results)
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Agent Regression Report - {agent_class_name}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                .summary {{ display: flex; gap: 20px; margin-bottom: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 6px; }}
                .summary-item {{ flex: 1; text-align: center; }}
                .summary-value {{ font-size: 24px; font-weight: bold; }}
                .summary-label {{ color: #666; font-size: 14px; }}
                .passed {{ color: #28a745; }}
                .failed {{ color: #dc3545; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; color: #333; font-weight: 600; }}
                tr:hover {{ background-color: #f1f1f1; }}
                .status-badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; text-transform: uppercase; }}
                .status-passed {{ background-color: #d4edda; color: #155724; }}
                .status-failed {{ background-color: #f8d7da; color: #721c24; }}
                .details-row {{ display: none; background-color: #fafafa; }}
                .details-content {{ padding: 15px; border-left: 4px solid #ddd; margin: 10px 0; }}
                .toggle-btn {{ cursor: pointer; color: #007bff; text-decoration: underline; border: none; background: none; padding: 0; }}
                pre {{ white-space: pre-wrap; word-wrap: break-word; background: #eee; padding: 10px; border-radius: 4px; }}
                .metric-badge {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px; margin-right: 5px; background-color: #e9ecef; color: #495057; border: 1px solid #ced4da; }}
                .metric-score {{ font-weight: bold; }}
                .model-id {{ font-size: 12px; color: #666; font-style: italic; }}
                /* Markdown styles */
                .markdown-content {{ background: #f9f9f9; padding: 15px; border-radius: 4px; border: 1px solid #eee; }}
                .markdown-content h1, .markdown-content h2, .markdown-content h3 {{ margin-top: 10px; margin-bottom: 10px; }}
                .markdown-content p {{ margin-bottom: 10px; }}
                .markdown-content code {{ background-color: #f0f0f0; padding: 2px 4px; border-radius: 3px; font-family: monospace; }}
                .markdown-content pre {{ background-color: #f0f0f0; padding: 10px; border-radius: 4px; overflow-x: auto; }}
                .markdown-content ul, .markdown-content ol {{ padding-left: 20px; margin-bottom: 10px; }}
                .markdown-content blockquote {{ border-left: 4px solid #ddd; padding-left: 10px; color: #666; margin: 10px 0; }}
            </style>
            <script>
                function toggleDetails(id) {{
                    var row = document.getElementById('details-' + id);
                    if (row.style.display === 'table-row') {{
                        row.style.display = 'none';
                    }} else {{
                        row.style.display = 'table-row';
                    }}
                }}
            </script>
        </head>
        <body>
            <div class="container">
                <h1>Agent Regression Report</h1>
                
                <div class="summary">
                    <div class="summary-item">
                        <div class="summary-value">{agent_class_name}</div>
                        <div class="summary-label">Agent Class</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">{total_count}</div>
                        <div class="summary-label">Total Scenarios</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value passed">{passed_count}</div>
                        <div class="summary-label">Passed</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value failed">{failed_count}</div>
                        <div class="summary-label">Failed</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">{pass_rate:.1f}%</div>
                        <div class="summary-label">Pass Rate</div>
                    </div>
                </div>

                <table>
                    <thead>
                        <tr>
                            <th style="width: 50px;">#</th>
                            <th>Agent Name</th>
                            <th>Scenario</th>
                            <th>Model</th>
                            <th>Status</th>
                            <th>Score</th>
                            <th>Metrics</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for i, res in enumerate(results):
            status_class = "status-passed" if res.get('passed') else "status-failed"
            status_text = "PASSED" if res.get('passed') else "FAILED"
            score = f"{res.get('score', 0):.1f}"
            agent_name = res.get('agent_name', 'Unknown')
            model_id = res.get('model_id', 'Unknown')
            
            # Format metrics for display
            metrics_html = ""
            if 'metrics' in res:
                for m_name, m_data in res['metrics'].items():
                    metrics_html += f'<span class="metric-badge">{m_name}: <span class="metric-score">{m_data["score"]}</span></span>'
            
            # Convert actual_output from markdown to HTML
            actual_output = res.get('actual_output', '')
            try:
                actual_output_html = markdown.markdown(actual_output, extensions=['fenced_code', 'tables'])
            except Exception:
                # Fallback if markdown conversion fails or markdown lib not available
                actual_output_html = f"<pre>{actual_output}</pre>"

            html_content += f"""
                        <tr>
                            <td>{i+1}</td>
                            <td>{agent_name}</td>
                            <td>{res.get('scenario', 'Unknown')}</td>
                            <td><span class="model-id">{model_id}</span></td>
                            <td><span class="status-badge {status_class}">{status_text}</span></td>
                            <td>{score}/10</td>
                            <td>{metrics_html}</td>
                            <td><button class="toggle-btn" onclick="toggleDetails({i})">View Details</button></td>
                        </tr>
                        <tr id="details-{i}" class="details-row">
                            <td colspan="8">
                                <div class="details-content">
                                    <strong>Input:</strong>
                                    <pre>{res.get('input', '')}</pre>
                                    
                                    <strong>Actual Output:</strong>
                                    <div class="markdown-content">{actual_output_html}</div>
                                    
                                    <strong>Expected Output:</strong>
                                    <pre>{res.get('expected_output', 'N/A')}</pre>
                                    
                                    <strong>Explanation:</strong>
                                    <pre>{res.get('explanation', '')}</pre>
                                    
                                    <strong>Token Usage:</strong>
                                    <pre>{json.dumps(res.get('token_usage') or {}, indent=2)}</pre>

                                    {f"<strong>Error:</strong><pre>{res.get('error')}</pre>" if res.get('error') else ""}
                                </div>
                            </td>
                        </tr>
            """

        html_content += """
                    </tbody>
                </table>
                
                <div style="margin-top: 20px; color: #666; font-size: 12px; text-align: center;">
                    Generated on """ + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
                </div>
            </div>
        </body>
        </html>
        """

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return filepath
