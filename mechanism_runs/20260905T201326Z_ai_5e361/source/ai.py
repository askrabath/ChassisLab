"""Two proposal operations, at most one repair per operation, no tool access."""
import json
from pydantic import create_model, ValidationError
from openai import OpenAI
from .schema import Design, Proposal

# Transport schema retains bounds but defers cross-field validation until after
# the response and usage have been saved. This preserves usage on geometry errors.
WireDesign=create_model('WireDesign',__config__=Design.model_config,
                        **{k:(f.annotation,f) for k,f in Design.model_fields.items()})
WireProposal=create_model('WireProposal',__config__=Proposal.model_config,
                          **{k:((WireDesign if k=='design' else f.annotation),f) for k,f in Proposal.model_fields.items()})


class ProposalFailure(RuntimeError):
    pass


class Designer:
    def __init__(self, model, save, context, client=None):
        self.model=model
        self.save=save
        self.context=context
        self.client=client or OpenAI(max_retries=0,timeout=60)
        self.calls=[]

    def propose(self, parent=None, results=None):
        operation='initial' if parent is None else 'revision'
        payload=dict(operation=operation,engineering_context=self.context,
                     parent=parent.model_dump() if parent else None,development_results=results)
        messages=[dict(role='system',content=(
            'Design one VEX-inspired four-wheel skid-steer chassis. Return a specification, concise rationale, '
            'predicted benefit, and a quantitative testable hypothesis. You may change only schema parameters. '
            'No claims of official legality or verified hardware performance. Use results honestly. '
            'Obey cross-field geometry constraints. Battery and optional ballast coordinates are meters.')),
            dict(role='user',content=json.dumps(payload))]
        for attempt in range(2):
            if len(self.calls)>=4: raise ProposalFailure('AI call budget exhausted')
            record=dict(operation=operation,attempt=attempt+1,model=self.model,
                        max_output_tokens=2500,input=messages.copy())
            self.calls.append(record)
            self.save('ai_calls.json',self.calls)
            try:
                response=self.client.responses.parse(model=self.model,input=messages,
                    text_format=WireProposal,max_output_tokens=2500,reasoning={'effort':'low'},store=False)
                record.update(response_id=response.id,status=response.status,
                              usage=response.usage.model_dump() if response.usage else None,
                              output=response.output_text)
                if response.output_parsed is None:
                    raise ValueError('No parsed output (refusal or incomplete response)')
                proposal=Proposal.model_validate(response.output_parsed.model_dump())
                record['validated']=True
                self.save('ai_calls.json',self.calls)
                return proposal
            except (ValidationError,ValueError) as exc:
                # Validation errors contain only proposal fields, never credentials.
                error=str(exc)[:2500]
                record.update(validated=False,validation_error=error)
                if record.get('output'):
                    messages.append(dict(role='assistant',content=record['output']))
                messages.append(dict(role='user',content='Repair the proposal. Validation failed: '+error))
            except Exception as exc:
                # Never serialize arbitrary transport exception text or request headers.
                record.update(validated=False,error_type=type(exc).__name__,
                              http_status=getattr(exc,'status_code',None))
                self.save('ai_calls.json',self.calls)
                raise ProposalFailure(f'API operation failed: {type(exc).__name__}; see ai_calls.json') from None
            self.save('ai_calls.json',self.calls)
        raise ProposalFailure('Proposal rejected after one repair; see ai_calls.json')
